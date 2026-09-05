//! `Session` and `Transaction` — canonical snapshot/working CoW
//! transaction model.
//!
//! The Python binding and the bolt-server (`backend.rs::TxState`) each grew
//! their own copy of this pattern; it lives here once so future bindings
//! don't multiply the drift.
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
    ///
    /// ## Why the fork takes a fresh `graph_id`
    ///
    /// A fork is the one clone that becomes an **independently mutable
    /// lineage**: the base keeps running and the working copy diverges from it.
    /// `graph_id` used to be cloned along with everything else, which made the
    /// Cypher plan cache's key — `(graph_id, version, …)` — ambiguous the
    /// moment two forks of one base each took a bump: identical key, different
    /// graphs. That is not merely a stale-statistics risk, because
    /// `fuse_anchored_edge_count` bakes a resolved physical `NodeIndex` into
    /// `Clause::FusedCountAnchoredEdges`; a sibling served that plan counts a
    /// different node's edges and returns a **wrong number** with no error (see
    /// `plan_cache_cost_tests::a_sibling_fork_is_never_served_another_forks_plan`).
    ///
    /// Minting here costs no cache reuse worth having. `working_mut` only
    /// reaches this path when the base Arc is still owned elsewhere — the
    /// genuinely-divergent case — and takes the `Arc::try_unwrap` *move* when it
    /// is not, which stays one lineage and rightly keeps its id. A fork's very
    /// first mutation bumps `version` anyway, so every plan the old id could
    /// still have matched was already unreachable by version alone.
    pub(crate) fn fork_transaction(&self) -> Self {
        let mut child = self.clone();
        child.graph_id = crate::graph::dir_graph::next_graph_id();
        child.graph.adopt_shared_writer_lineage(&self.graph);
        child
    }
}

/// Shared graph state. Sessions live in bindings' top-level state
/// (Python's `KnowledgeGraph.inner` is conceptually a Session; the
/// bolt-server's `KgliteBackend.session` IS one).
///
/// **Concurrency model.** The outer `Mutex` is brief-acquire-only:
/// - [`snapshot`](Self::snapshot) takes the lock, `Arc::clone`s the
///   inner, releases.
/// - [`commit`](Self::commit) takes the lock to swap the inner Arc
///   with the new (post-mutation) DirGraph. Readers holding old Arc
///   clones keep their stable view across the swap.
///
/// Bindings that need cross-session coordination (bolt-server's
/// per-session tx state) layer their own `Arc<Mutex<...>>` over
/// the Session. The Session itself is `Send + Sync`.
pub struct Session {
    pub(super) graph: Mutex<Arc<DirGraph>>,
    /// Write-ahead-log state for a session opened via
    /// [`Session::open_durable`]; `None` for every ordinary session.
    ///
    /// It lives here rather than on `DirGraph` because it owns an open `File`
    /// and `DirGraph` must stay `Clone`. **The lock is only ever taken while
    /// [`graph`](Self::graph)'s lock is already held, or on its own — never
    /// the other way round.** That ordering is what makes "append the frame,
    /// then publish the Arc" indivisible for concurrent committers. See
    /// [`super::durable`].
    pub(super) durable: Mutex<Option<super::durable::DurableState>>,
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
    pub fn new(graph: DirGraph) -> Self {
        Self::from_arc(Arc::new(graph))
    }

    /// Construct from a graph already shared by Arc (e.g. wrapping
    /// `KnowledgeGraph.inner`).
    ///
    /// The session is **not** durable; see [`Session::open_durable`] for the
    /// write-ahead-logged constructor.
    pub fn from_arc(graph: Arc<DirGraph>) -> Self {
        Self {
            graph: Mutex::new(graph),
            durable: Mutex::new(None),
        }
    }

    /// Take a snapshot of the current graph. Wait-free apart from the momentary
    /// mutex acquire, and poison-recovering rather than cascading another
    /// thread's panic: the snapshot is just an Arc clone, so consistency is
    /// about the next reader's Arc value, not the inner DirGraph.
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
    ///
    /// # Not for durable sessions
    ///
    /// A session opened via [`Session::open_durable`] **must not** take this
    /// path — its mutations are captured by the recording backend but nothing
    /// drains that buffer into a WAL frame, and the next copy-on-write fork
    /// resets it, so the write would apply and then vanish on a crash. Callers
    /// with an error channel check [`Session::check_direct_write_allowed`]
    /// first.
    ///
    /// This signature has none: it returns a guard, and a guard cannot carry a
    /// refusal. Silently dropping the write is the very hazard being closed, so
    /// the refusal is **latched** instead — the durable session records that the
    /// log no longer describes the graph, and every subsequent durability
    /// operation fails loudly: [`commit`](Self::commit) returns
    /// [`CommitOutcome::DurabilityFailed`] and [`sync`](Session::sync) errors,
    /// until a [`save`](Self::save) checkpoint folds the direct write in and
    /// starts a fresh log. The mutation is never lost and never mistaken for a
    /// logged one.
    pub fn write(&self) -> SessionWriteGuard<'_> {
        self.mark_diverged();
        SessionWriteGuard {
            guard: self.graph.lock().unwrap_or_else(|p| p.into_inner()),
        }
    }

    /// Run a detached serialized transaction under one Session lock. The
    /// closure sees a transaction fork; success swaps it atomically and bumps
    /// the live version once, while error drops it with no partial writes.
    ///
    /// **A batch that wrote nothing is a no-op**, mirroring
    /// [`CommitOutcome::NoWritesNoOp`] on the [`commit`](Self::commit) path:
    /// no version bump, no Arc swap, the fork is dropped. An unnecessary bump
    /// is exactly what makes a concurrent OCC committer lose a race it should
    /// win: the other transaction's `base_version` no longer matches, and it is
    /// told to retry against a graph that never changed.
    ///
    /// The write test is the version delta on the fork, because
    /// [`DirGraph::bump_version`] is the canonical "this graph just mutated"
    /// signal that every mutation path routes through. Detection is
    /// *post*-execution: the fork still happens, because the closure needs a
    /// `&mut DirGraph` and the atomicity guarantee (an error publishes
    /// nothing) depends on mutating a copy. That fork is O(changes), and
    /// skipping the swap is what the caller actually observes.
    ///
    /// # Not for durable sessions
    ///
    /// Like [`write`](Self::write), and for the same reason: the closure's
    /// mutations are captured but never drained into a WAL frame. The refusal
    /// cannot travel in-band either — `E` is the *caller's* error type and the
    /// engine cannot construct one — so this path latches the session exactly
    /// as `write` does, and every later durability operation fails until a
    /// checkpoint repairs it. Callers that own an error channel check
    /// [`Session::check_direct_write_allowed`] before calling.
    pub fn transact<T, E>(
        &self,
        operation: impl FnOnce(&mut DirGraph) -> Result<T, E>,
    ) -> Result<T, E> {
        self.mark_diverged();
        let mut guard = self.graph.lock().unwrap_or_else(|p| p.into_inner());
        let current_version = guard.version();
        let mut working = guard.fork_transaction();
        let value = operation(&mut working)?;
        if working.version() == current_version {
            return Ok(value);
        }
        working.set_version(current_version + 1);
        *guard = Arc::new(working);
        Ok(value)
    }

    /// Checkpoint the session's current graph to `path`.
    ///
    /// Routes through the shared [`save_graph_with`](crate::graph::io::file::save_graph_with)
    /// dispatch, so mode-aware save (disk directory vs `.kgl`), atomic
    /// temp+rename, the recorded storage mode, and the `fsync` barrier behave
    /// exactly as for a binding that owns its `Arc<DirGraph>` directly.
    ///
    /// **This exists because saving through [`snapshot`](Self::snapshot)
    /// cannot be done cheaply.** A save *mutates* the graph it writes — save
    /// metadata, index keys, columnar consolidation — so a caller holding a
    /// snapshot clone hands `Arc::make_mut` a shared pointer and deep-clones
    /// every node, edge and index on every checkpoint. The session's own Arc
    /// is behind a private mutex, so only the session can reach the
    /// unique-owner path; a binding that holds a `Session` has no other route
    /// to a no-copy save.
    ///
    /// The session lock is held for the write, which is what makes the
    /// checkpoint a consistent point-in-time image: concurrent writers and
    /// *new* snapshots wait, while readers already holding a snapshot are
    /// unaffected. The save is not a semantic mutation and does not bump the
    /// graph version, so an in-flight OCC transaction is undisturbed.
    ///
    /// This is a persistence decision, not a write-ownership claim. A caller
    /// that may publish to `path` must hold its own
    /// [`GraphWriterLease`](crate::graph::io::open::GraphWriterLease) across
    /// the whole open/mutate/save interval.
    ///
    /// # Durable sessions
    ///
    /// For a session opened via [`Session::open_durable`] this method is the
    /// **checkpoint**, and it runs the four-step order the format requires:
    /// flush the log → stamp `checkpoint_lsn` into the graph being written →
    /// write the `.kgl` → drop the capture buffer and truncate the log. The
    /// `graph::session::durable` module documents why each step sits where it
    /// does. Saving to a different destination transfers future logging there
    /// after successful publication and preserves the original recovery log.
    /// The caller must hold the destination writer lease before save-as and
    /// retain it for the remaining session lifetime, as with open_durable.
    ///
    /// `fsync=false` is **overridden to true** on a durable session, silently
    /// and deliberately: the checkpoint destroys the log that would otherwise
    /// still describe those commits, so pairing a maybe-not-on-disk checkpoint
    /// with a destroyed log would let one crash lose both. (The Python wheel
    /// warns as well as overriding; the engine has no warning channel, so the
    /// contract is stated here.)
    pub fn save(&self, path: &str, fsync: bool) -> Result<(), String> {
        let mut guard = self.graph.lock().unwrap_or_else(|p| p.into_inner());
        self.save_checkpoint(&mut guard, path, fsync)
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

    /// Commit a transaction. Returns a [`CommitOutcome`] so the binding can
    /// map it to its own error type.
    ///
    /// OCC is opt-in: pass `true` for `check_occ` to enforce, `false` for
    /// last-writer-wins — which silently discards any write that landed
    /// between this tx's begin and commit.
    pub fn commit(&self, tx: Transaction, check_occ: bool) -> CommitOutcome {
        let (working_opt, base_version) = tx.take_working();
        let Some(mut working) = working_opt else {
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

        // Durable sessions log the commit BEFORE publishing it. A frame that
        // could not be appended means the caller must not be told the write
        // happened, so the Arc swap below is skipped entirely — the working
        // copy is dropped and the graph is exactly as it was. (The wheel's
        // apply-then-log ordering can only report the same failure *after* the
        // mutation is visible; this is the stronger half of the rung.) No-op
        // for a non-durable session.
        if let Err(error) = self.log_working_commit(&mut working) {
            return CommitOutcome::DurabilityFailed { error };
        }

        // Bump from the *current* version (not the possibly-stale base) so the
        // version is monotonic even in last-writer-wins mode (check_occ=false).
        let new_version = current_version + 1;
        working.set_version(new_version);
        *guard = Arc::new(working);
        // Assignment drops the former owner before checking layer ownership.
        // Retained snapshots still prevent their shared bases from being folded.
        if let Some(published) = Arc::get_mut(&mut guard) {
            crate::graph::handle::compact_dir_graph(published);
        }
        CommitOutcome::Committed { new_version }
    }

    /// Roll back a transaction. The working copy (if materialized)
    /// is dropped; no Arc swap. Cannot fail.
    pub fn rollback(&self, _tx: Transaction) {
        // Drop `_tx`: the snapshot Arc decrements and the working copy is freed.
    }
}

/// Snapshot/working CoW transaction state.
///
/// **State machine**:
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
    /// Whether this tx was opened read-only via [`Session::begin_read`].
    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    /// Graph version at BEGIN time, for a binding that runs its own OCC check
    /// outside [`Session::commit`].
    pub fn base_version(&self) -> u64 {
        self.base_version
    }

    /// Whether this tx has materialized a working copy (a mutation has fired).
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
///
/// `#[non_exhaustive]`: a commit can fail in ways that did not exist when a
/// binding was written — [`DurabilityFailed`](Self::DurabilityFailed) is the
/// first — and an outcome a binding does not recognise must reach its error
/// path, not its success path. Downstream matches carry a catch-all arm.
#[derive(Debug)]
#[non_exhaustive]
pub enum CommitOutcome {
    /// Read-only-then-commit / no mutations happened. Cheap path.
    NoWritesNoOp,
    /// Working copy was swapped into the shared graph. The new version is one
    /// past the *current* version under the commit lock, which equals this tx's
    /// `base_version` only when no other writer intervened.
    Committed { new_version: u64 },
    /// OCC conflict: another writer committed between this tx's
    /// `begin` and `commit`. The current shared graph's version is
    /// `current_version`; this tx's base was `base_version`. The
    /// working copy is dropped (lost).
    ConflictDetected {
        current_version: u64,
        base_version: u64,
    },
    /// **Durable sessions only.** The commit's write-ahead-log frame could not
    /// be appended, so the commit was not published: the graph is unchanged,
    /// the version did not move, and the working copy is dropped (lost).
    ///
    /// This is a hard failure, not a retriable conflict — the disk, not
    /// another writer, said no. A binding surfaces it as an IO/backend error;
    /// re-running the unit of work will hit the same wall until the underlying
    /// problem is cleared.
    DurabilityFailed { error: String },
}

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

    /// The property the whole method exists for: a checkpoint writes through
    /// the session's own Arc. Asserted with the backend clone counter rather
    /// than by inspecting the file, because a snapshot-based save produces a
    /// byte-identical file — it just copies the entire graph to get there,
    /// silently, on every save.
    #[test]
    fn save_writes_through_the_sessions_own_graph_without_cloning() {
        use crate::graph::session::execute::{execute_mut, execute_read, ExecuteOptions};
        use crate::graph::storage::backend::{backend_clone_count, reset_backend_clone_count};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("checkpoint.kgl");
        let path_str = path.to_string_lossy().into_owned();
        let params = std::collections::HashMap::new();
        let opts = ExecuteOptions::eager(&params);

        let session = Session::new(empty_graph());
        execute_mut(&mut session.write(), "CREATE (:N {id: 1})", &opts).unwrap();

        reset_backend_clone_count();
        session.save(&path_str, false).unwrap();
        assert_eq!(
            backend_clone_count(),
            0,
            "a checkpoint must write through the Session's own Arc; saving a \
             snapshot() clone deep-clones every node, edge and index"
        );

        // …and the file carries the write, so the no-clone path is not a
        // no-op path.
        let reloaded = crate::graph::io::file::load_file(&path_str).unwrap();
        let outcome = execute_read(&reloaded, "MATCH (n:N) RETURN n.id AS id", &opts).unwrap();
        assert_eq!(
            outcome.result.rows.len(),
            1,
            "the checkpoint must contain the node created before it"
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

    /// **T5 — a batch that writes nothing must not advance the version.**
    ///
    /// Red before the fix on every assertion here: `transact` forked,
    /// `set_version(current + 1)` and swapped unconditionally, so an empty
    /// batch and a read-only batch each reported version 1 and published a new
    /// Arc. The sibling `commit` path has always answered `NoWritesNoOp` for
    /// exactly this case; the two paths now agree.
    #[test]
    fn transact_with_no_writes_is_a_noop() {
        let s = Session::new(empty_graph());
        let before = s.snapshot();

        // Empty closure — the `kglite_create_edges_batch([])` shape.
        let value = s.transact(|_working| Ok::<_, &'static str>(7)).unwrap();
        assert_eq!(value, 7);
        assert_eq!(s.version(), 0, "an empty batch must not bump the version");
        assert!(
            Arc::ptr_eq(&before, &s.snapshot()),
            "an empty batch must not swap the live Arc"
        );

        // Read-only closure — the `execute_mut_batch` of MATCH statements
        // shape. `execute_mut` only bumps when the statement is a mutation, so
        // the fork's version is the signal.
        let value = s
            .transact(|working| {
                let _ = working.version();
                Ok::<_, &'static str>(8)
            })
            .unwrap();
        assert_eq!(value, 8);
        assert_eq!(s.version(), 0);
        assert!(Arc::ptr_eq(&before, &s.snapshot()));

        // Non-vacuity: a batch that *does* write still commits, exactly once.
        s.transact(|working| {
            working.bump_version();
            working.bump_version();
            Ok::<_, &'static str>(())
        })
        .unwrap();
        assert_eq!(s.version(), 1, "one writing transaction is one version");
        assert!(!Arc::ptr_eq(&before, &s.snapshot()));
    }

    /// The reason the no-op matters: an interleaved empty batch used to make a
    /// concurrent OCC committer lose a race it should win. `other` begins at
    /// version V, an unrelated empty batch runs, `other` commits — with the
    /// spurious bump its `base_version` no longer matched and it was told to
    /// retry against a graph nothing had changed.
    #[test]
    fn an_empty_batch_does_not_make_a_concurrent_committer_conflict() {
        let s = Session::new(empty_graph());
        let mut other = s.begin();
        other.working_mut().unwrap().bump_version();

        s.transact(|_working| Ok::<_, &'static str>(())).unwrap();

        assert!(
            matches!(
                s.commit(other, /* check_occ = */ true),
                CommitOutcome::Committed { new_version: 1 }
            ),
            "an empty batch must not invalidate a concurrent transaction"
        );

        // Non-vacuity: a real interleaved write still conflicts.
        let mut loser = s.begin();
        loser.working_mut().unwrap().bump_version();
        s.transact(|working| {
            working.bump_version();
            Ok::<_, &'static str>(())
        })
        .unwrap();
        assert!(matches!(
            s.commit(loser, true),
            CommitOutcome::ConflictDetected { .. }
        ));
    }

    /// `add_edges_from_specs` bumped the version at the end of the function
    /// regardless of what it had done, so an empty spec list bumped inside the
    /// fork too — which `transact`'s version-delta test would then have read
    /// as a real write.
    #[test]
    fn an_empty_edge_spec_batch_writes_nothing_and_bumps_nothing() {
        use crate::graph::mutation::maintain::add_edges_from_specs;

        let s = Session::new(empty_graph());
        let before = s.snapshot();
        let report = s
            .transact(|working| add_edges_from_specs(working, Vec::new()))
            .unwrap();
        assert_eq!(report.connections_created, 0);
        assert_eq!(report.skipped_missing_endpoint, 0);
        assert_eq!(s.version(), 0);
        assert!(Arc::ptr_eq(&before, &s.snapshot()));
    }

    #[test]
    fn begin_then_commit_no_writes_is_noop() {
        let s = Session::new(empty_graph());
        let tx = s.begin();
        let outcome = s.commit(tx, /* check_occ = */ true);
        assert!(matches!(outcome, CommitOutcome::NoWritesNoOp));
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

        let mut tx_a = s.begin();
        let _ = tx_a.working_mut().unwrap();

        let mut tx_b = s.begin();
        let _ = tx_b.working_mut().unwrap();
        let outcome_b = s.commit(tx_b, true);
        assert!(matches!(
            outcome_b,
            CommitOutcome::Committed { new_version: 1 }
        ));

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
        assert!(!Arc::ptr_eq(&pre, &post));
    }

    #[test]
    fn double_commit_via_take_working_drops_state() {
        // `commit` takes the Transaction by value and drains it through
        // `take_working`, so a second commit of the same state is
        // unrepresentable rather than merely discouraged. Two things that are
        // observable, and would break if the drain ever became a copy: the
        // working copy moves out exactly once, and a drained (write-less) tx
        // commits as a no-op that leaves the version where it was.
        let s = Session::new(empty_graph());
        let mut tx = s.begin();
        tx.working_mut().expect("rw tx");
        let (working, base) = tx.take_working();
        assert!(
            working.is_some(),
            "the materialized working copy moves to the caller"
        );
        assert_eq!(base, 0);
        assert_eq!(
            s.version(),
            0,
            "taking the working copy commits nothing by itself"
        );

        let drained = Transaction {
            snapshot: None,
            working: None,
            base_version: base,
            read_only: false,
        };
        assert!(
            matches!(s.commit(drained, true), CommitOutcome::NoWritesNoOp),
            "a tx whose working copy is gone has nothing left to apply"
        );
        assert_eq!(s.version(), 0, "and so must not bump the version");
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
                                other => {
                                    panic!("materialised tx must commit as a write: {other:?}")
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
