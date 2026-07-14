//! `Session` and `Transaction` — canonical snapshot/working CoW
//! transaction model.
//!
//! Mirrors the pattern that previously lived inline in
//! `src/graph/pyapi/transaction.rs` (Python-bound) and was mirrored
//! again in `crates/kglite-bolt-server/src/backend.rs::TxState`
//! (per-Bolt-session). Phase E extracts it once so future bindings
//! (Go, TypeScript, JVM) don't multiply the drift.
//!
//! ## Shape
//!
//! - [`Session`] owns the shared `Arc<DirGraph>` behind a `Mutex`
//!   so commits can atomically swap the inner Arc.
//! - [`Transaction`] holds an optional `snapshot: Arc<DirGraph>`
//!   taken at BEGIN time + an optional `working: DirGraph`
//!   materialized lazily on first mutation. Memory/mapped transactions clone
//!   their stable BEGIN snapshot; disk transactions remap immutable bases and
//!   copy only mutation overlays.
//! - [`Session::commit`] performs the OCC version check + Arc
//!   swap; returns [`CommitOutcome`] so the binding decides how to
//!   surface conflicts to its consumers.

use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::error::KgError;
use crate::graph::dir_graph::DirGraph;

impl DirGraph {
    /// Create a transaction working copy. Disk backends remap immutable base
    /// arrays and inherit the serialized writer lineage; memory/mapped modes
    /// retain their ordinary snapshot clone semantics.
    pub(crate) fn fork_transaction(&self) -> Self {
        let mut child = self.clone();
        child.graph.adopt_transaction_lineage(&self.graph);
        child
    }
}

/// Shared graph state. Sessions live in bindings' top-level state
/// (Python's `KnowledgeGraph.inner` is conceptually a Session; the
/// bolt-server's `KgliteBackend.graph` IS one).
///
/// **Concurrency model.** The outer `Mutex` is brief-acquire-only:
/// - [`snapshot`](Self::snapshot) takes the lock, `Arc::clone`s the
///   inner, releases. Readers see a stable graph view via their
///   Arc<DirGraph> handle that survives subsequent commits.
/// - [`commit`](Self::commit) takes the lock to swap the inner Arc
///   with the new (post-mutation) DirGraph. Readers holding old
///   Arc clones keep their stable view.
///
/// Bindings that need cross-session coordination (bolt-server's
/// per-session tx state) layer their own `Arc<Mutex<...>>` over
/// the Session. The Session itself is `Send + Sync`.
pub struct Session {
    /// Inner Arc allows cheap reader snapshots; outer Mutex allows
    /// atomic commit-swap.
    graph: Mutex<Arc<DirGraph>>,
}

/// Serialized mutable access to a Session graph. The guard holds the Session
/// mutex for the complete write, so a uniquely-owned Arc mutates in place and
/// a held reader snapshot triggers copy-on-write exactly once.
pub struct SessionWriteGuard<'a> {
    guard: MutexGuard<'a, Arc<DirGraph>>,
}

impl Deref for SessionWriteGuard<'_> {
    type Target = DirGraph;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl DerefMut for SessionWriteGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        if Arc::get_mut(&mut self.guard).is_none() {
            let child = self.guard.fork_transaction();
            *self.guard = Arc::new(child);
        }
        Arc::get_mut(&mut self.guard).expect("Session write guard owns the active graph")
    }
}

impl Session {
    /// Construct from an owned DirGraph.
    pub fn new(graph: DirGraph) -> Self {
        Self {
            graph: Mutex::new(Arc::new(graph)),
        }
    }

    /// Construct from an existing Arc<DirGraph>. Used when the
    /// caller already shares the graph via Arc (e.g. wrapping
    /// `KnowledgeGraph.inner`).
    pub fn from_arc(graph: Arc<DirGraph>) -> Self {
        Self {
            graph: Mutex::new(graph),
        }
    }

    /// Take a snapshot of the current graph. Wait-free apart from
    /// the momentary mutex acquire. Poison-recovers — a panic in
    /// another thread that left the mutex poisoned doesn't cascade
    /// here; we accept the inconsistent state and continue. (The
    /// snapshot itself is just an Arc clone; consistency is about
    /// the next reader's Arc value, not the inner DirGraph.)
    pub fn snapshot(&self) -> Arc<DirGraph> {
        Arc::clone(&self.graph.lock().unwrap_or_else(|p| p.into_inner()))
    }

    /// Current graph version. Reads through a snapshot so the
    /// mutex hold is brief. Used by bindings for OCC checks
    /// without going through [`begin`](Self::begin).
    pub fn version(&self) -> u64 {
        self.snapshot().version()
    }

    /// Lock the Session for one serialized mutation and return its mutable
    /// graph view. Unlike `begin()` this does not first clone the Session's
    /// Arc, so the unique mutable path is reachable when no reader snapshot
    /// is alive. Readers that already hold a snapshot remain on the
    /// prior graph; new snapshots wait for this short write guard.
    pub fn write(&self) -> SessionWriteGuard<'_> {
        SessionWriteGuard {
            guard: self.graph.lock().unwrap_or_else(|p| p.into_inner()),
        }
    }

    /// Run a detached serialized transaction under one Session lock. The
    /// closure sees a transaction fork; success swaps it atomically and bumps
    /// the live version once, while error drops it with no partial writes.
    pub fn transact<T, E>(
        &self,
        operation: impl FnOnce(&mut DirGraph) -> Result<T, E>,
    ) -> Result<T, E> {
        let mut guard = self.graph.lock().unwrap_or_else(|p| p.into_inner());
        let current_version = guard.version();
        let mut working = guard.fork_transaction();
        let value = operation(&mut working)?;
        working.set_version(current_version + 1);
        *guard = Arc::new(working);
        Ok(value)
    }

    /// Begin a new read-write transaction. The snapshot is taken
    /// immediately; the working copy is deferred until the first
    /// mutation (see [`Transaction::working_mut`]).
    pub fn begin(&self) -> Transaction {
        let snapshot = self.snapshot();
        let base_version = snapshot.version();
        Transaction {
            snapshot: Some(snapshot),
            working: None,
            base_version,
            read_only: false,
        }
    }

    /// Begin a read-only transaction. Mutations through
    /// [`Transaction::working_mut`] return `KgError::Argument`.
    pub fn begin_read(&self) -> Transaction {
        let mut tx = self.begin();
        tx.read_only = true;
        tx
    }

    /// Commit a transaction. Returns the outcome so the binding
    /// can map to its error type:
    /// - [`CommitOutcome::NoWritesNoOp`] — tx didn't mutate; no
    ///   Arc swap, no version bump. Cheap.
    /// - [`CommitOutcome::Committed { new_version }`] — working
    ///   was swapped into the shared graph. `new_version` reflects
    ///   the bumped value (`base_version + 1`).
    /// - [`CommitOutcome::ConflictDetected`] — another writer
    ///   committed between this tx's `begin` and `commit` (the
    ///   shared graph's version > `tx.base_version`). Binding
    ///   typically surfaces this as a typed retry-suggesting error.
    ///
    /// OCC is opt-in: pass `true` for `check_occ` to enforce. Pass
    /// `false` for last-writer-wins semantics (current bolt-server
    /// default until the binding wires the check).
    pub fn commit(&self, tx: Transaction, check_occ: bool) -> CommitOutcome {
        let (working_opt, base_version) = tx.take_working();
        let Some(mut working) = working_opt else {
            // Read-only-then-commit / no mutations — Arc swap not
            // needed.
            return CommitOutcome::NoWritesNoOp;
        };

        // Hold ONE lock guard across both the OCC check and the Arc swap so
        // check-and-swap is atomic. Reading the version via `self.version()`
        // (which locks, clones, unlocks) and then swapping under a *separate*
        // lock acquisition is a TOCTOU race: two concurrent committers could
        // both pass the check and both swap — losing one commit and even
        // moving the version backwards. The Python `Session` masks this with a
        // writer lock (one committer at a time), but the core `Session` is
        // driven concurrently by the bolt-server, so the atomicity must live
        // here. (std `Mutex` is not reentrant, so read the version off the
        // guarded Arc, never via `self.version()`.)
        let mut guard = self.graph.lock().unwrap_or_else(|p| p.into_inner());
        let current_version = guard.version();
        if check_occ && current_version != base_version {
            return CommitOutcome::ConflictDetected {
                current_version,
                base_version,
            };
        }

        // Bump from the *current* version (not the possibly-stale base) so the
        // version is monotonic even in last-writer-wins mode (check_occ=false).
        let new_version = current_version + 1;
        working.set_version(new_version);
        *guard = Arc::new(working);
        CommitOutcome::Committed { new_version }
    }

    /// Roll back a transaction. The working copy (if materialized)
    /// is dropped; no Arc swap. Cannot fail.
    pub fn rollback(&self, _tx: Transaction) {
        // Drop _tx → snapshot Arc count decrements; working
        // DirGraph (if Some) is freed. Nothing else to do.
    }
}

/// Snapshot/working CoW transaction state.
///
/// **State machine** (mirrors `src/graph/pyapi/transaction.rs`):
///
/// - **Initial / read-only-after-begin**: `snapshot: Some, working:
///   None`. Reads route through `snapshot`. No clone cost.
/// - **After first mutation**: `snapshot: None, working: Some`. The
///   snapshot Arc is consumed. An owned snapshot can move directly; otherwise
///   the backend-specific transaction fork preserves isolation. Reads
///   and writes both route through `working`.
/// - **After commit / rollback**: `snapshot: None, working: None`.
///   Calls to `current()` or `working_mut()` fail with
///   `KgError::Argument`.
pub struct Transaction {
    pub(super) snapshot: Option<Arc<DirGraph>>,
    pub(super) working: Option<DirGraph>,
    pub(super) base_version: u64,
    pub(super) read_only: bool,
}

impl Transaction {
    /// Whether this tx was opened read-only via
    /// [`Session::begin_read`]. Read-only txs reject
    /// [`working_mut`](Self::working_mut) calls.
    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    /// Graph version at BEGIN time. Used by the binding for its
    /// own OCC checks if it wants to do them outside
    /// [`Session::commit`].
    pub fn base_version(&self) -> u64 {
        self.base_version
    }

    /// Whether this tx has materialized a working copy (first
    /// mutation has fired).
    pub fn has_writes(&self) -> bool {
        self.working.is_some()
    }

    /// Current graph view. Prefer this for reads inside the tx —
    /// returns the working copy if materialized, else the snapshot.
    /// Returns `None` only after commit/rollback (defensive; should
    /// not happen with correct caller use).
    pub fn current(&self) -> Option<&DirGraph> {
        self.working.as_ref().or(self.snapshot.as_deref())
    }

    /// Materialize the working copy if needed and return `&mut
    /// DirGraph` for mutation. Reads via [`current`](Self::current)
    /// after the first mutation route through the same working
    /// copy automatically.
    ///
    /// Rejected with `KgError::Argument` if:
    /// - The tx is read-only (`begin_read`).
    /// - The tx has been committed/rolled back (no snapshot, no
    ///   working).
    // KgError carries transaction context; boxing it would only burden an error path.
    #[allow(clippy::result_large_err)]
    pub fn working_mut(&mut self) -> Result<&mut DirGraph, KgError> {
        if self.read_only {
            return Err(KgError::Argument(
                "read-only transaction does not support mutations \
                 (CREATE/SET/DELETE/REMOVE/MERGE) — open a read-write tx \
                 via Session::begin"
                    .to_string(),
            ));
        }
        if self.working.is_none() {
            let snap = self.snapshot.take().ok_or_else(|| {
                KgError::Argument("transaction already committed or rolled back".to_string())
            })?;
            // Move an unusually unique snapshot directly; normal Session/KG
            // transactions retain an owner Arc and therefore use the
            // backend-specific transaction fork.
            let working = Arc::try_unwrap(snap).unwrap_or_else(|arc| arc.fork_transaction());
            self.working = Some(working);
        }
        Ok(self
            .working
            .as_mut()
            .expect("invariant: just materialized above"))
    }

    /// Consume the transaction. Returns `(working, base_version)`.
    /// `working` is `Some` iff [`working_mut`](Self::working_mut)
    /// was called at least once. Used by [`Session::commit`].
    pub fn take_working(self) -> (Option<DirGraph>, u64) {
        (self.working, self.base_version)
    }
}

/// Outcome of [`Session::commit`]. Bindings inspect this to decide
/// what to surface to their consumers.
#[derive(Debug)]
pub enum CommitOutcome {
    /// Read-only-then-commit / no mutations happened. Cheap path.
    NoWritesNoOp,
    /// Working copy was swapped into the shared graph. The new
    /// version is `base_version + 1`.
    Committed { new_version: u64 },
    /// OCC conflict: another writer committed between this tx's
    /// `begin` and `commit`. The current shared graph's version is
    /// `current_version`; this tx's base was `base_version`. The
    /// working copy is dropped (lost).
    ConflictDetected {
        current_version: u64,
        base_version: u64,
    },
}

// ────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_graph() -> DirGraph {
        DirGraph::new()
    }

    #[test]
    fn new_session_version_is_zero() {
        let s = Session::new(empty_graph());
        assert_eq!(s.version(), 0);
    }

    #[test]
    fn snapshot_is_cheap_arc_clone() {
        let s = Session::new(empty_graph());
        let snap1 = s.snapshot();
        let snap2 = s.snapshot();
        // Both Arcs point at the same inner DirGraph.
        assert!(Arc::ptr_eq(&snap1, &snap2));
    }

    #[test]
    fn serialized_write_uses_unique_arc_in_place() {
        use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};

        for mode in [StorageMode::Memory, StorageMode::Mapped] {
            let s = Session::new(new_dir_graph_in_mode(mode, None).unwrap());
            let before = {
                let guard = s.graph.lock().unwrap();
                Arc::as_ptr(&guard)
            };
            {
                let mut graph = s.write();
                graph.bump_version();
            }
            let after = {
                let guard = s.graph.lock().unwrap();
                Arc::as_ptr(&guard)
            };
            assert_eq!(
                before, after,
                "unique Session write must not clone the graph"
            );
            assert_eq!(s.version(), 1);
        }
    }

    #[test]
    fn serialized_execute_skips_checkpoint_for_proven_single_node_create() {
        use crate::graph::session::execute::{execute_mut, ExecuteOptions};
        use crate::graph::storage::backend::{backend_clone_count, reset_backend_clone_count};

        let params = std::collections::HashMap::new();
        let mut opts = ExecuteOptions::eager(&params);
        opts.deadline = Some(std::time::Instant::now() + std::time::Duration::from_secs(60));

        let unique = Session::new(empty_graph());
        reset_backend_clone_count();
        execute_mut(&mut unique.write(), "CREATE (:N {id: 1})", &opts).unwrap();
        assert_eq!(
            backend_clone_count(),
            0,
            "proven single-node CREATE must not clone a uniquely-owned graph"
        );

        let shared = Session::new(empty_graph());
        let _reader = shared.snapshot();
        reset_backend_clone_count();
        execute_mut(&mut shared.write(), "CREATE (:N {id: 1})", &opts).unwrap();
        assert_eq!(
            backend_clone_count(),
            1,
            "held reader needs only the working fork, not a second checkpoint"
        );

        let checkpointed = Session::new(empty_graph());
        reset_backend_clone_count();
        execute_mut(
            &mut checkpointed.write(),
            "CREATE (:N {id: 1}), (:N {id: 2})",
            &opts,
        )
        .unwrap();
        assert_eq!(
            backend_clone_count(),
            1,
            "multi-element CREATE must retain its atomic rollback checkpoint"
        );

        reset_backend_clone_count();
        execute_mut(
            &mut checkpointed.write(),
            "MATCH (n:N {id: 1}) DELETE n",
            &opts,
        )
        .unwrap();
        assert_eq!(
            backend_clone_count(),
            0,
            "terminal preflighted DELETE must not clone the graph"
        );
    }

    #[test]
    fn serialized_write_forks_when_reader_snapshot_is_held() {
        let s = Session::new(empty_graph());
        let old = s.snapshot();
        {
            let mut graph = s.write();
            graph.bump_version();
        }
        let current = s.snapshot();
        assert!(!Arc::ptr_eq(&old, &current));
        assert_eq!(old.version(), 0);
        assert_eq!(current.version(), 1);
    }

    #[test]
    fn serialized_transaction_swaps_once_or_discards_on_error() {
        let s = Session::new(empty_graph());
        let old = s.snapshot();
        let value = s
            .transact(|working| {
                working.bump_version();
                working.bump_version();
                Ok::<_, &'static str>(42)
            })
            .unwrap();
        assert_eq!(value, 42);
        assert_eq!(s.version(), 1, "one transaction is one committed version");
        assert_eq!(old.version(), 0);

        let failed = s.transact(|working| {
            working.bump_version();
            Err::<(), _>("cancelled")
        });
        assert_eq!(failed, Err("cancelled"));
        assert_eq!(
            s.version(),
            1,
            "failed transaction must not reach the live Arc"
        );
    }

    #[test]
    fn begin_then_commit_no_writes_is_noop() {
        let s = Session::new(empty_graph());
        let tx = s.begin();
        let outcome = s.commit(tx, /* check_occ = */ true);
        assert!(matches!(outcome, CommitOutcome::NoWritesNoOp));
        // Version unchanged.
        assert_eq!(s.version(), 0);
    }

    #[test]
    fn begin_then_rollback_is_noop() {
        let s = Session::new(empty_graph());
        let tx = s.begin();
        s.rollback(tx);
        assert_eq!(s.version(), 0);
    }

    #[test]
    fn working_mut_materializes_only_on_first_call() {
        let s = Session::new(empty_graph());
        let mut tx = s.begin();
        assert!(!tx.has_writes());
        assert!(tx.current().is_some());
        // First working_mut materializes.
        let _ = tx.working_mut().unwrap();
        assert!(tx.has_writes());
        assert!(tx.snapshot.is_none());
        assert!(tx.working.is_some());
    }

    #[test]
    fn current_routes_through_working_after_materialize() {
        let s = Session::new(empty_graph());
        let mut tx = s.begin();
        let _ = tx.working_mut().unwrap();
        // current() now returns &working, not &snapshot.
        let _: &DirGraph = tx.current().unwrap();
    }

    #[test]
    fn commit_with_writes_bumps_version() {
        let s = Session::new(empty_graph());
        let mut tx = s.begin();
        let _ = tx.working_mut().unwrap();
        let outcome = s.commit(tx, /* check_occ = */ true);
        match outcome {
            CommitOutcome::Committed { new_version } => assert_eq!(new_version, 1),
            other => panic!("expected Committed, got {other:?}"),
        }
        assert_eq!(s.version(), 1);
    }

    #[test]
    fn read_only_tx_rejects_working_mut() {
        let s = Session::new(empty_graph());
        let mut tx = s.begin_read();
        assert!(tx.is_read_only());
        match tx.working_mut() {
            Err(KgError::Argument(msg)) => assert!(msg.contains("read-only")),
            Err(other) => panic!("expected Argument, got different error: {other}"),
            Ok(_) => panic!("expected read-only rejection but got Ok"),
        }
    }

    #[test]
    fn read_only_tx_commit_is_noop() {
        let s = Session::new(empty_graph());
        let tx = s.begin_read();
        let outcome = s.commit(tx, /* check_occ = */ true);
        assert!(matches!(outcome, CommitOutcome::NoWritesNoOp));
        assert_eq!(s.version(), 0);
    }

    #[test]
    fn occ_conflict_detected_when_other_writer_commits() {
        let s = Arc::new(Session::new(empty_graph()));

        // Tx A: begins, mutates, doesn't commit yet.
        let mut tx_a = s.begin();
        let _ = tx_a.working_mut().unwrap();

        // Tx B: begins (sees version 0), mutates, commits → version 1.
        let mut tx_b = s.begin();
        let _ = tx_b.working_mut().unwrap();
        let outcome_b = s.commit(tx_b, true);
        assert!(matches!(
            outcome_b,
            CommitOutcome::Committed { new_version: 1 }
        ));

        // Tx A: commits → conflict (base_version=0, current=1).
        let outcome_a = s.commit(tx_a, true);
        match outcome_a {
            CommitOutcome::ConflictDetected {
                current_version,
                base_version,
            } => {
                assert_eq!(current_version, 1);
                assert_eq!(base_version, 0);
            }
            other => panic!("expected ConflictDetected, got {other:?}"),
        }
        // The shared graph still reflects B's commit.
        assert_eq!(s.version(), 1);
    }

    #[test]
    fn occ_disabled_means_last_writer_wins() {
        let s = Arc::new(Session::new(empty_graph()));

        let mut tx_a = s.begin();
        let _ = tx_a.working_mut().unwrap();
        let mut tx_b = s.begin();
        let _ = tx_b.working_mut().unwrap();

        // Without OCC, both commits succeed; B's data wins (it swaps last).
        let outcome_a = s.commit(tx_a, /* check_occ = */ false);
        let outcome_b = s.commit(tx_b, /* check_occ = */ false);
        assert!(matches!(outcome_a, CommitOutcome::Committed { .. }));
        assert!(matches!(outcome_b, CommitOutcome::Committed { .. }));
        // Two commits → version 2, monotonic. Each commit bumps from the
        // *current* version under the lock (0→1→2), NOT from the tx's
        // (possibly stale) base_version. Monotonicity is required for OCC
        // soundness: "version changed ⇒ graph changed" must hold, so two
        // changes must yield two distinct versions even in last-writer-wins
        // mode. (The prior behaviour bumped from base_version, leaving both at
        // 1 — a latent bug where a later OCC tx could miss B's change.)
        assert_eq!(s.version(), 2);
    }

    #[test]
    fn snapshot_after_commit_sees_new_graph() {
        let s = Session::new(empty_graph());
        let pre = s.snapshot();
        assert_eq!(pre.version(), 0);

        let mut tx = s.begin();
        let _ = tx.working_mut().unwrap();
        let _ = s.commit(tx, true);

        let post = s.snapshot();
        assert_eq!(post.version(), 1);
        // The two snapshots are different Arcs (commit replaced
        // the inner).
        assert!(!Arc::ptr_eq(&pre, &post));
    }

    #[test]
    fn double_commit_via_take_working_drops_state() {
        // The current API takes Transaction by value in commit, so
        // double-commit is statically impossible (the second call
        // doesn't have a tx to pass). Pin this invariant via a
        // compile-time check by-construction. (A previous Python
        // boundary used an Option<...> field and raised at runtime;
        // the value-take API improves on that.)
        let s = Session::new(empty_graph());
        let tx = s.begin();
        let _ = s.commit(tx, true);
        // Cannot call s.commit(tx, true) again — tx was moved.
    }

    // ── True-parallel concurrency tests ─────────────────────────────────
    //
    // Unlike the Python-level Session stress tests (which the GIL partly
    // serialises), these drive the core `Session` from real OS threads with
    // no GIL — so they exercise genuine parallel access to the
    // `Mutex<Arc<DirGraph>>`, the snapshot/commit Arc-swap, and the OCC
    // version check. They are the intended targets for `cargo +nightly test
    // -Z sanitizer=thread` (see docs/rust/concurrency-verification.md): a data
    // race in the locking/commit path surfaces here, not in single-threaded
    // tests.

    #[test]
    fn concurrent_writers_compose_with_occ_retry() {
        // N threads each commit `per` times via begin → mutate → commit with
        // OCC, retrying on conflict. Every commit must land exactly once:
        // final version == N*per. A lost commit (racey Arc-swap or version
        // bump) would show as version < N*per; a double-apply as version >.
        const N: u64 = 8;
        const PER: u64 = 200;
        let session = Arc::new(Session::new(empty_graph()));
        let handles: Vec<_> = (0..N)
            .map(|_| {
                let s = Arc::clone(&session);
                std::thread::spawn(move || {
                    for _ in 0..PER {
                        loop {
                            let mut tx = s.begin();
                            // Materialise a working copy so the commit counts
                            // as a write (bumps version + swaps).
                            tx.working_mut().expect("rw tx");
                            match s.commit(tx, /* check_occ = */ true) {
                                CommitOutcome::Committed { .. } => break,
                                CommitOutcome::ConflictDetected { .. } => continue,
                                CommitOutcome::NoWritesNoOp => {
                                    panic!("materialised tx must commit as a write")
                                }
                            }
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("worker thread panicked");
        }
        assert_eq!(
            session.version(),
            N * PER,
            "OCC-retried commits must compose with no lost or doubled updates"
        );
    }

    #[test]
    fn occ_detects_conflict_between_overlapping_txs() {
        // Two txs from the same base version: the first commits, the second
        // must be told it conflicts (its base is now stale).
        let s = Session::new(empty_graph());
        let mut tx1 = s.begin();
        tx1.working_mut().unwrap();
        let mut tx2 = s.begin();
        tx2.working_mut().unwrap();
        assert!(matches!(
            s.commit(tx1, true),
            CommitOutcome::Committed { .. }
        ));
        assert!(matches!(
            s.commit(tx2, true),
            CommitOutcome::ConflictDetected { .. }
        ));
        assert_eq!(s.version(), 1, "only the winning commit bumped the version");
    }

    #[test]
    fn concurrent_snapshots_consistent_under_commits() {
        // Readers take snapshots + read the version while writers commit. A
        // snapshot's version must always be a committed value (0..=total) and
        // monotonically non-decreasing per reader — the Arc swap is atomic.
        const WRITERS: u64 = 4;
        const PER: u64 = 250;
        const READERS: usize = 4;
        let total = WRITERS * PER;
        let session = Arc::new(Session::new(empty_graph()));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let readers: Vec<_> = (0..READERS)
            .map(|_| {
                let s = Arc::clone(&session);
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    let mut last = 0u64;
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        let snap = s.snapshot();
                        let v = snap.version();
                        assert!(v <= total, "snapshot version {v} exceeds total {total}");
                        assert!(v >= last, "version went backwards: {v} < {last}");
                        last = v;
                    }
                })
            })
            .collect();

        let writers: Vec<_> = (0..WRITERS)
            .map(|_| {
                let s = Arc::clone(&session);
                std::thread::spawn(move || {
                    for _ in 0..PER {
                        loop {
                            let mut tx = s.begin();
                            tx.working_mut().unwrap();
                            if matches!(s.commit(tx, true), CommitOutcome::Committed { .. }) {
                                break;
                            }
                        }
                    }
                })
            })
            .collect();

        for h in writers {
            h.join().unwrap();
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for h in readers {
            h.join().unwrap();
        }
        assert_eq!(session.version(), total);
    }
}
