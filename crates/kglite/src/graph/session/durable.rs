//! Durable sessions — the write-ahead log wired through [`Session`].
//!
//! The engine carries a complete logical WAL (`graph/wal.rs`:
//! CRC frames, torn-tail recovery, [`DurabilityLevel`], the `checkpoint_lsn`
//! replay gate) plus the write-capture layer that feeds it
//! (`graph/storage/recording.rs`). What lived *only* in the Python wheel was
//! the orchestration: the open ordering, the commit-time flush, and the
//! four-step checkpoint. This module lifts that orchestration into the engine
//! so every binding — wheel, bolt server, C ABI, JVM — gets it from one place
//! instead of writing it again.
//!
//! ## The three orders that are correctness, not preference
//!
//! Two of them are not session-specific and live in
//! [`graph::durability`](crate::graph::durability), which this module calls and
//! the Python wheel's durable `KnowledgeGraph` calls too — the open ordering
//! (including the unconditional recovery-on-open refusal) and the two halves of
//! the checkpoint. Only the second order below belongs to `Session`, because it
//! is a property of how a session publishes a commit.
//!
//! 1. **Open: recover → replay → wrap → open-for-append.** Replay must happen
//!    *before* the backend is wrapped for capture, or the replay's own
//!    `GraphWrite` calls land in the capture buffer and get logged a second
//!    time. Replay is gated on the loaded graph's `checkpoint_lsn`, so frames
//!    already folded into the `.kgl` are skipped rather than rolled back over
//!    newer state. ([`durability::open_log`](crate::graph::durability::open_log))
//!
//! 2. **Commit: append *then* publish.** [`Session::commit`] appends the
//!    frame between the OCC check and the `Arc` swap, so a failed append
//!    blocks the publish and the caller is told
//!    ([`CommitOutcome::DurabilityFailed`](super::CommitOutcome::DurabilityFailed))
//!    instead of holding an acknowledged-but-unlogged write. This is strictly
//!    stronger than the wheel's apply-then-log ordering, which can only report
//!    the failure after the mutation is already visible.
//!
//! 3. **Checkpoint: sync → stamp → save → reset.** [`Session::save`] flushes
//!    the log (load-bearing under [`DurabilityLevel::Normal`], where the tail
//!    may still be in the page cache), stamps `checkpoint_lsn = next_lsn - 1`
//!    into the graph it is about to write, writes the `.kgl`, and only then
//!    truncates the log. `next_lsn` deliberately keeps climbing across the
//!    truncation: restarting it would let a stale pre-checkpoint frame carry
//!    the same LSN as a fresh one, making the replay gate meaningless.
//!    ([`durability::checkpoint_prologue`](crate::graph::durability::checkpoint_prologue)
//!    and
//!    [`checkpoint_epilogue`](crate::graph::durability::checkpoint_epilogue))
//!
//! ## Lock ordering
//!
//! `DurableState` lives on the [`Session`] behind its own `Mutex`, not on
//! `DirGraph` — the WAL owns a `File` handle and `DirGraph` must stay `Clone`.
//! **The durability mutex is only ever acquired while the session's graph
//! mutex is already held** (or on its own, never the other way round), which
//! is what makes "append the frame, then publish the Arc" one indivisible
//! step for concurrent committers.
//!
//! ## Single owner per path
//!
//! One `DurableState` per checkpoint path, full stop. Two owners logging to
//! one sidecar interleave their independent `next_lsn` counters and each
//! checkpoint stamps a `checkpoint_lsn` the other's frames sit below, so
//! replay silently drops committed data. The engine enforces what it can
//! observe: [`Session::open_durable`] refuses a graph whose backend is
//! *already* wrapped for capture, which is the signature another durable owner
//! leaves behind. Making the wheel's `KnowledgeGraph` and a `Session` mutually
//! exclusive over one path is the binding layer's half of the same rule.

use std::path::Path;
use std::sync::{Arc, Mutex};

use super::transaction::Session;
use crate::graph::dir_graph::DirGraph;
use crate::graph::durability;
use crate::graph::storage::recording::resolve_ops;
use crate::graph::storage::GraphRead;
use crate::graph::wal::{DurabilityLevel, Wal, WalFrame};

/// Session-scoped durability state. Held by the [`Session`] rather than the
/// graph because it owns an open `File`; see the module docs for the lock
/// ordering it participates in.
#[derive(Debug)]
pub(super) struct DurableState {
    wal: Wal,
    /// Log-sequence number the next frame will carry. Monotonic for the life
    /// of the log; never reset by a checkpoint.
    next_lsn: u64,
    /// The level the caller asked for. Never [`DurabilityLevel::Off`] — an
    /// `Off` session has no `DurableState` at all.
    level: DurabilityLevel,
    /// Set when a mutation reached the graph through a path the log cannot
    /// describe ([`Session::write`] / [`Session::transact`]). See
    /// [`Session::write`] for why this is a latch rather than a refusal.
    diverged: bool,
    /// Test-only append fault injection. The guarantee under test — a failed
    /// append blocks the publish — is about *ordering*, and ordering cannot be
    /// exercised without a reachable append failure; no portable filesystem
    /// trick fails a write on an already-open append handle.
    #[cfg(test)]
    fail_append: bool,
}

/// Message shared by every operation that a direct write has invalidated.
const DIVERGED_MSG: &str = "this durable session was mutated through Session::write / \
     Session::transact, which the write-ahead log cannot describe: those mutations are \
     captured but never drained into a frame, so the log no longer describes the graph. \
     Take a checkpoint (Session::save) to fold them in and start a fresh log, or run \
     mutations through Session::begin / Session::commit, which are logged.";

/// Message shared by the two direct-write paths.
pub(super) const DIRECT_WRITE_REFUSAL: &str =
    "a durable session does not support direct writes through Session::write / \
     Session::transact: their mutations are captured by the recording backend but \
     nothing drains that buffer into a WAL frame, and the next copy-on-write fork \
     resets it — the write would apply and then vanish on a crash, with no error. \
     Run mutations through Session::begin / Session::commit, which append a frame \
     before publishing.";

impl Session {
    /// Open `graph` as a **durable session** checkpointed at `checkpoint_path`,
    /// performing the full recover → replay → wrap → open-for-append ordering
    /// described in the module docs.
    ///
    /// `checkpoint_path` is the `.kgl` path, not the log path; the sidecar is
    /// derived from it by [`wal_path`] exactly as every other binding derives
    /// it, so a graph saved by one and reopened by another finds the same log.
    ///
    /// # Levels
    ///
    /// [`DurabilityLevel::Full`] and [`DurabilityLevel::Normal`] produce a
    /// logging session. [`DurabilityLevel::Off`] produces an ordinary
    /// non-durable session — **unless** the sidecar still holds frames this
    /// graph has not folded in, in which case the call is refused rather than
    /// silently opening a graph that is missing committed writes (the first
    /// checkpoint would then truncate them away for good).
    ///
    /// # Refusals
    ///
    /// - **Disk-mode graphs**, at any logging level: a disk graph commits by
    ///   publishing an immutable generation, so there is no logical WAL for it
    ///   at any level.
    /// - **A graph already wrapped for capture**: that is the observable
    ///   signature of another durable owner over the same data, and two owners
    ///   sharing one log corrupt each other's replay gate (module docs).
    /// - **Unreplayed frames at level `Off`**, as above.
    ///
    /// # Ownership
    ///
    /// Pass a graph you solely own. A shared `Arc` is deep-cloned here (the
    /// replay and the capture wrap both need `&mut DirGraph`), and the session
    /// then owns the copy while the other holder keeps the original — two
    /// divergent graphs, only one of them logged. The caller must also hold
    /// its own
    /// [`GraphWriterLease`](crate::graph::io::open::GraphWriterLease) over
    /// `checkpoint_path` for the life of the session: a durable session both
    /// appends to the sidecar and republishes the checkpoint.
    pub fn open_durable(
        graph: Arc<DirGraph>,
        checkpoint_path: &str,
        level: DurabilityLevel,
    ) -> Result<Session, String> {
        if level.logs() && graph.graph.is_disk() {
            return Err(format!(
                "durable={} is not supported for storage='disk' (only 'off' is). A disk \
                 graph commits by publishing an immutable generation, so its durability \
                 boundary is the generation publish, not a logical write-ahead log: a \
                 replayed WAL frame and a published generation can each describe the same \
                 commit, and reconciling them needs a generation-aware log this release \
                 does not have. Use Session::save checkpoints for disk graphs, or a \
                 mapped / in-memory graph if you need per-commit crash safety.",
                level.name(),
            ));
        }

        let mut graph = graph;
        // Everything from here to the open log is the shared orchestration
        // (`graph::durability`): recover → replay → wrap → open-for-append, plus
        // the unconditional recovery-on-open refusal that makes `off` over an
        // unreplayed sidecar an error rather than silent data loss. The wheel's
        // durable `KnowledgeGraph` performs the same sequence through the same
        // function, so the two cannot drift.
        let opened = durability::open_log(&mut graph, Path::new(checkpoint_path), level)
            .map_err(|e| e.to_string())?;

        let Some((wal, next_lsn)) = opened else {
            return Ok(Session::from_arc(graph));
        };

        Ok(Session::with_durable(
            graph,
            DurableState {
                wal,
                next_lsn,
                level,
                diverged: false,
                #[cfg(test)]
                fail_append: false,
            },
        ))
    }

    /// The durability level this session logs at, or `None` for an ordinary
    /// non-durable session (including one opened with
    /// [`DurabilityLevel::Off`], which carries no log at all).
    pub fn durability(&self) -> Option<DurabilityLevel> {
        self.durable
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .map(|ds| ds.level)
    }

    /// Flush every committed frame to stable storage — the barrier
    /// [`DurabilityLevel::Full`] takes on every commit, taken on demand.
    ///
    /// This is what makes [`DurabilityLevel::Normal`] adoptable rather than
    /// merely fast: without it, a `Normal` session's only route to
    /// power-safety is a full checkpoint, which republishes the entire graph —
    /// the wrong granularity for "flush at the end of a request".
    ///
    /// Under `Full` every frame is already on stable storage when `commit`
    /// returns, so this is a no-op. On a non-durable session it is an error
    /// rather than a silent success: there is no log to flush, so reporting
    /// "flushed" would be a lie about the only thing the caller asked.
    pub fn sync(&self) -> Result<(), String> {
        // Graph lock first, then the durability lock — the module's ordering
        // rule, kept even though this path touches only the latter.
        let _graph = self.graph.lock().unwrap_or_else(|p| p.into_inner());
        let mut slot = self.durable.lock().unwrap_or_else(|p| p.into_inner());
        let Some(ds) = slot.as_mut() else {
            return Err(
                "sync() needs a session opened with a write-ahead log (Session::open_durable \
                 at level 'full' or 'normal'). This session has none, so there is nothing to \
                 flush and no power-safe point to take — call Session::save to write a \
                 checkpoint instead."
                    .to_string(),
            );
        };
        if ds.diverged {
            return Err(DIVERGED_MSG.to_string());
        }
        if ds.level == DurabilityLevel::Full {
            return Ok(());
        }
        ds.wal.sync().map_err(|e| e.to_string())
    }

    /// Append `working`'s captured mutations as one WAL frame. Called by
    /// [`Session::commit`] with the session's graph lock already held, between
    /// the OCC check and the `Arc` swap, so an error here means the commit is
    /// never published.
    ///
    /// A no-op on a non-durable session, and on a transaction that captured
    /// nothing: an empty frame would consume an LSN and describe no change,
    /// which is exactly the accounting the checkpoint stamp reads.
    pub(super) fn log_working_commit(&self, working: &mut DirGraph) -> Result<(), String> {
        let mut slot = self.durable.lock().unwrap_or_else(|p| p.into_inner());
        let Some(ds) = slot.as_mut() else {
            // Not durable — but this is still the commit boundary, and it is
            // where a change-data-capture log gets its events (and where the
            // capture buffer of a CDC-only graph is discarded, so it stays
            // bounded). A no-op on a graph carrying no capture layer.
            crate::graph::cdc::drain_at_commit(working);
            return Ok(());
        };
        if ds.diverged {
            return Err(DIVERGED_MSG.to_string());
        }
        // A durable session wraps its graph at open and every fork preserves
        // the wrapper, so an unwrapped backend here means the capture seam was
        // lost and this commit is unloggable. Fail closed rather than publish
        // an unlogged write.
        let raw = match working.graph.recording_mut() {
            Some(rg) => rg.take_ops(),
            None => {
                return Err(
                    "durable commit found no recording backend: this session's write-capture \
                     layer was replaced, so the commit cannot be logged and has not been \
                     published."
                        .to_string(),
                )
            }
        };
        if raw.is_empty() {
            return Ok(());
        }
        // Secondary labels are read back through `working` because they are not
        // backend state — see `resolve_ops`.
        let ops = resolve_ops(&raw, &working.graph, &working.interner, |idx| {
            working.secondary_label_names(idx)
        });
        #[cfg(test)]
        if ds.fail_append {
            return Err("injected WAL append failure".to_string());
        }
        let lsn = ds.next_lsn;
        ds.wal
            .append(&WalFrame { lsn, ops })
            .map_err(|e| e.to_string())?;
        // Only a frame that reached the log consumes its LSN.
        ds.next_lsn = lsn + 1;
        // Published from the same drained ops the frame was built from, so the
        // two views of this commit cannot disagree about what it changed — and
        // *after* the append, because an append failure aborts the commit
        // (`Session::commit` skips the `Arc` swap): publishing first would put
        // a change into the stream that the caller was told did not happen.
        crate::graph::cdc::publish_drained(working, &raw);
        Ok(())
    }

    /// Checkpoint steps 1–2: flush the log, then stamp how far this checkpoint
    /// will have consumed it into the graph about to be serialized. Returns
    /// whether this session is durable (i.e. whether the epilogue runs).
    pub(super) fn checkpoint_prologue(&self, graph: &mut Arc<DirGraph>) -> Result<bool, String> {
        let mut slot = self.durable.lock().unwrap_or_else(|p| p.into_inner());
        let Some(ds) = slot.as_mut() else {
            return Ok(false);
        };
        // Flush-then-stamp, shared with every other owner of a log; see
        // `durability::checkpoint_prologue` for why the flush is load-bearing
        // under `Normal` and how the stamp is derived.
        durability::checkpoint_prologue(&mut ds.wal, ds.next_lsn, Arc::make_mut(graph))
            .map_err(|e| e.to_string())?;
        Ok(true)
    }

    /// Checkpoint step 4: the `.kgl` now holds the full current state, so drop
    /// the capture buffer (those ops are folded in) and truncate the log.
    ///
    /// Ordered after the save on purpose: the checkpoint is known to have
    /// landed before the log that describes the same commits is destroyed, and
    /// replay is idempotent, so a crash between the two costs only a harmless
    /// re-apply on the next open.
    pub(super) fn checkpoint_epilogue(&self, graph: &mut Arc<DirGraph>) -> Result<(), String> {
        let mut slot = self.durable.lock().unwrap_or_else(|p| p.into_inner());
        let Some(ds) = slot.as_mut() else {
            return Ok(());
        };
        durability::checkpoint_epilogue(&mut ds.wal, Arc::make_mut(graph))
            .map_err(|e| e.to_string())?;
        // The checkpoint just folded in everything a direct write left
        // unlogged, so the log describes the graph again.
        ds.diverged = false;
        Ok(())
    }

    /// Record that a mutation reached the graph through a path the log cannot
    /// describe. See [`Session::write`] for the reasoning behind the latch.
    pub(super) fn mark_diverged(&self) {
        if let Some(ds) = self
            .durable
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_mut()
        {
            ds.diverged = true;
        }
    }

    /// Whether direct mutation through [`Session::write`] / [`Session::transact`]
    /// is allowed on this session, as a checkable precondition.
    ///
    /// A durable session answers `Err` with the message those paths would
    /// otherwise have to swallow — neither of them has an error channel of its
    /// own (`write` returns a guard, `transact` returns the *caller's* error
    /// type, which the engine cannot construct). A caller that owns an error
    /// channel calls this first; a caller that does not still cannot lose data
    /// silently, because both paths latch the session (see [`Session::write`]).
    pub fn check_direct_write_allowed(&self) -> Result<(), String> {
        if self
            .durable
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .is_some()
        {
            return Err(DIRECT_WRITE_REFUSAL.to_string());
        }
        Ok(())
    }

    /// Inject a write-ahead-append failure. Test-only, and `pub(crate)`
    /// rather than `pub(super)` because the durability failure path is also
    /// what the change-data-capture tests use to prove a refused commit
    /// publishes nothing.
    #[cfg(test)]
    pub(crate) fn set_fail_append(&self, fail: bool) {
        if let Some(ds) = self
            .durable
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_mut()
        {
            ds.fail_append = fail;
        }
    }

    #[cfg(test)]
    pub(super) fn next_lsn(&self) -> Option<u64> {
        self.durable
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .map(|ds| ds.next_lsn)
    }

    /// Build a session that already owns durability state. Private to the
    /// session module: [`Session::open_durable`] is the only constructor that
    /// can establish the recover→replay→wrap ordering the state assumes.
    fn with_durable(graph: Arc<DirGraph>, state: DurableState) -> Session {
        Session {
            graph: Mutex::new(graph),
            durable: Mutex::new(Some(state)),
        }
    }
}

// ────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::session::execute::{execute_mut, execute_read, ExecuteOptions};
    use crate::graph::session::CommitOutcome;
    use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};
    use crate::graph::wal::{recover, wal_path};
    use std::collections::HashMap;

    fn params() -> HashMap<String, crate::datatypes::Value> {
        HashMap::new()
    }

    /// Run one mutation through the begin → mutate → commit path, i.e. the
    /// only write path a durable session supports.
    fn commit_query(session: &Session, query: &str) -> CommitOutcome {
        let params = params();
        let opts = ExecuteOptions::eager(&params);
        let mut tx = session.begin();
        execute_mut(tx.working_mut().unwrap(), query, &opts).unwrap();
        session.commit(tx, /* check_occ = */ true)
    }

    /// The refusal message from a rejected `open_durable`. A helper rather
    /// than `expect_err` because `Session` deliberately carries no `Debug`
    /// impl — printing a session would print the whole graph.
    fn refusal(result: Result<Session, String>) -> String {
        match result {
            Err(message) => message,
            Ok(_) => panic!("expected a refusal, got an open session"),
        }
    }

    fn count_nodes(graph: &DirGraph) -> usize {
        let params = params();
        let opts = ExecuteOptions::eager(&params);
        execute_read(graph, "MATCH (n:N) RETURN n.id AS id", &opts)
            .unwrap()
            .result
            .rows
            .len()
    }

    fn fresh(path: &std::path::Path) -> Session {
        Session::open_durable(
            Arc::new(DirGraph::new()),
            &path.to_string_lossy(),
            DurabilityLevel::Full,
        )
        .unwrap()
    }

    /// Reopen the way a *crash* reopens: load whatever checkpoint exists (none
    /// on the first run) and let `open_durable` replay the sidecar.
    fn reopen(path: &std::path::Path, level: DurabilityLevel) -> Result<Session, String> {
        let p = path.to_string_lossy().into_owned();
        let graph = if path.exists() {
            crate::graph::io::file::load_file(&p).unwrap()
        } else {
            Arc::new(DirGraph::new())
        };
        Session::open_durable(graph, &p, level)
    }

    // ── the headline guarantee ──────────────────────────────────────

    /// Crash-shaped: commits, no checkpoint, process gone. The reopen must
    /// replay every committed write out of the sidecar.
    #[test]
    fn committed_writes_replay_after_a_crash_shaped_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.kgl");

        let session = fresh(&path);
        assert!(matches!(
            commit_query(&session, "CREATE (:N {id: 1})"),
            CommitOutcome::Committed { .. }
        ));
        assert!(matches!(
            commit_query(&session, "CREATE (:N {id: 2})"),
            CommitOutcome::Committed { .. }
        ));
        assert_eq!(count_nodes(&session.snapshot()), 2);
        drop(session); // no save() — the checkpoint never happened

        assert!(!path.exists(), "the crash-shaped run wrote no checkpoint");
        let recovered = reopen(&path, DurabilityLevel::Full).unwrap();
        assert_eq!(
            count_nodes(&recovered.snapshot()),
            2,
            "both committed writes must come back out of the log"
        );
        // Replay must not have re-entered the capture buffer: the next commit
        // logs its own op and nothing else.
        assert_eq!(recovered.next_lsn(), Some(3));
        // …which is the replay-before-wrap rule, checked at its own seam.
        // Wrapping first would leave every replayed op sitting in the capture
        // buffer, ready to be logged a second time.
        match recovered.snapshot().graph.recording() {
            Some(rg) => assert_eq!(
                rg.ops_len(),
                0,
                "replay ran before the capture wrap, so it captured nothing"
            ),
            None => panic!("a durable session must be wrapped for capture"),
        }
    }

    /// **A failed append must block the publish.** Mutation-checked: moving
    /// the `log_working_commit` call after the `Arc` swap in `Session::commit`
    /// turns every assertion below red.
    #[test]
    fn a_failed_append_blocks_the_publish() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.kgl");
        let session = fresh(&path);

        commit_query(&session, "CREATE (:N {id: 1})");
        let version_before = session.version();
        let snapshot_before = session.snapshot();

        session.set_fail_append(true);
        let outcome = commit_query(&session, "CREATE (:N {id: 2})");
        match outcome {
            CommitOutcome::DurabilityFailed { ref error } => {
                assert!(error.contains("injected"), "unexpected error: {error}")
            }
            other => panic!("expected DurabilityFailed, got {other:?}"),
        }
        assert_eq!(
            session.version(),
            version_before,
            "a commit that could not be logged must not bump the version"
        );
        assert!(
            Arc::ptr_eq(&snapshot_before, &session.snapshot()),
            "a commit that could not be logged must not swap the live Arc"
        );
        assert_eq!(
            count_nodes(&session.snapshot()),
            1,
            "the unlogged write must not be visible"
        );

        // Non-vacuity: with the fault cleared the same write commits and lands.
        session.set_fail_append(false);
        assert!(matches!(
            commit_query(&session, "CREATE (:N {id: 2})"),
            CommitOutcome::Committed { .. }
        ));
        assert_eq!(count_nodes(&session.snapshot()), 2);
    }

    /// The four-step checkpoint, checked at each of its four observable
    /// consequences.
    #[test]
    fn save_truncates_the_log_and_stamps_the_replay_gate() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.kgl");
        let wpath = wal_path(&path);
        let session = fresh(&path);

        commit_query(&session, "CREATE (:N {id: 1})");
        commit_query(&session, "CREATE (:N {id: 2})");
        let logged_len = std::fs::metadata(&wpath).unwrap().len();
        assert!(logged_len > 0);

        session.save(&path.to_string_lossy(), true).unwrap();

        // (a) the log is truncated back to its header …
        let after_len = std::fs::metadata(&wpath).unwrap().len();
        assert!(
            after_len < logged_len,
            "checkpoint must truncate the log: {logged_len} -> {after_len}"
        );
        assert!(recover(&wpath).unwrap().is_empty());
        // (b) … the checkpoint carries the gate …
        let reloaded = crate::graph::io::file::load_file(&path.to_string_lossy()).unwrap();
        assert_eq!(reloaded.checkpoint_lsn, 2, "next_lsn(3) - 1");
        assert_eq!(count_nodes(&reloaded), 2);
        // (c) … and `next_lsn` keeps climbing across the truncation.
        assert_eq!(session.next_lsn(), Some(3));

        // (d) post-checkpoint commits land in the fresh log, and a
        // crash-shaped reopen replays only those.
        commit_query(&session, "CREATE (:N {id: 3})");
        drop(session);
        let frames = recover(&wpath).unwrap();
        assert_eq!(frames.len(), 1, "only the post-checkpoint commit is logged");
        assert_eq!(frames[0].lsn, 3);

        let recovered = reopen(&path, DurabilityLevel::Full).unwrap();
        assert_eq!(
            count_nodes(&recovered.snapshot()),
            3,
            "checkpointed 2 + replayed 1"
        );
    }

    /// The replay gate is not decorative: a sidecar whose frames the
    /// checkpoint already contains must not be folded in a second time. (Here
    /// the second fold is harmless-by-idempotence for counts, so the check is
    /// that `open_durable` skips the stale prefix — `next_lsn` proves it.)
    #[test]
    fn a_stale_prefix_below_the_checkpoint_is_not_replayed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.kgl");
        let wpath = wal_path(&path);

        let session = fresh(&path);
        commit_query(&session, "CREATE (:N {id: 1})");
        let logged = std::fs::read(&wpath).unwrap();
        session.save(&path.to_string_lossy(), true).unwrap();
        drop(session);
        // Restore the pre-checkpoint sidecar — the "operator put the backup
        // back" / "crash between save and truncate" shape.
        std::fs::write(&wpath, &logged).unwrap();

        let recovered = reopen(&path, DurabilityLevel::Full).unwrap();
        assert_eq!(count_nodes(&recovered.snapshot()), 1);
        assert_eq!(
            recovered.next_lsn(),
            Some(2),
            "the stale frame must not advance the log-sequence counter"
        );
    }

    // ── refusals ────────────────────────────────────────────────────

    #[test]
    fn level_off_refuses_an_unreplayed_sidecar_but_accepts_a_stale_one() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.kgl");
        let wpath = wal_path(&path);

        let session = fresh(&path);
        commit_query(&session, "CREATE (:N {id: 1})");
        drop(session);

        // Unreplayed frames + level off ⇒ refuse, naming the way out.
        let err = refusal(reopen(&path, DurabilityLevel::Off));
        assert!(err.contains("'full' or 'normal'"), "message was: {err}");
        assert!(err.contains("move the sidecar aside"), "message was: {err}");

        // Same sidecar, but now folded into a checkpoint: level off is fine.
        let session = reopen(&path, DurabilityLevel::Full).unwrap();
        let logged = std::fs::read(&wpath).unwrap();
        session.save(&path.to_string_lossy(), true).unwrap();
        drop(session);
        std::fs::write(&wpath, &logged).unwrap();
        let plain = reopen(&path, DurabilityLevel::Off).unwrap();
        assert!(plain.durability().is_none());
        assert_eq!(count_nodes(&plain.snapshot()), 1);
    }

    #[test]
    fn an_empty_sidecar_is_not_a_refusal_at_level_off() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.kgl");
        let session = reopen(&path, DurabilityLevel::Off).unwrap();
        assert!(session.durability().is_none());
    }

    #[test]
    fn disk_graphs_are_refused_at_every_logging_level() {
        let dir = tempfile::tempdir().unwrap();
        let graph_dir = dir.path().join("disk-graph");
        for level in [DurabilityLevel::Full, DurabilityLevel::Normal] {
            let g = new_dir_graph_in_mode(StorageMode::Disk, Some(&graph_dir)).unwrap();
            let err = refusal(Session::open_durable(
                Arc::new(g),
                &dir.path().join("g.kgl").to_string_lossy(),
                level,
            ));
            assert!(err.contains("storage='disk'"), "message was: {err}");
            assert!(err.contains(level.name()), "message was: {err}");
        }
        // Non-vacuity: `off` over a disk graph is a plain session, not an error.
        let g = new_dir_graph_in_mode(StorageMode::Disk, Some(&graph_dir)).unwrap();
        let session = Session::open_durable(
            Arc::new(g),
            &dir.path().join("g.kgl").to_string_lossy(),
            DurabilityLevel::Off,
        )
        .unwrap();
        assert!(session.durability().is_none());
    }

    #[test]
    fn a_second_durable_owner_over_one_graph_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.kgl");
        let session = fresh(&path);
        commit_query(&session, "CREATE (:N {id: 1})");

        // The observable signature of the first owner: the graph is wrapped.
        let err = refusal(Session::open_durable(
            session.snapshot(),
            &path.to_string_lossy(),
            DurabilityLevel::Full,
        ));
        assert!(err.contains("already wrapped"), "message was: {err}");
    }

    #[test]
    fn mapped_graphs_are_durable_too() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.kgl");
        let g = new_dir_graph_in_mode(StorageMode::Mapped, None).unwrap();
        let session = Session::open_durable(
            Arc::new(g),
            &path.to_string_lossy(),
            DurabilityLevel::Normal,
        )
        .unwrap();
        assert_eq!(session.durability(), Some(DurabilityLevel::Normal));
        commit_query(&session, "CREATE (:N {id: 1})");
        session.sync().unwrap();
        drop(session);

        let recovered = reopen(&path, DurabilityLevel::Normal).unwrap();
        assert_eq!(count_nodes(&recovered.snapshot()), 1);
    }

    #[test]
    fn open_durable_reports_an_unopenable_log() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.kgl");
        // Make the sidecar path a directory — a real IO failure inside
        // `Wal::open`, not an injected one.
        std::fs::create_dir(wal_path(&path)).unwrap();
        let err = refusal(Session::open_durable(
            Arc::new(DirGraph::new()),
            &path.to_string_lossy(),
            DurabilityLevel::Full,
        ));
        assert!(
            err.contains("write-ahead log"),
            "IO failure must name the log: {err}"
        );
    }

    // ── direct-write refusal ────────────────────────────────────────

    #[test]
    fn direct_writes_are_refused_and_latch_the_session() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.kgl");
        let session = fresh(&path);
        commit_query(&session, "CREATE (:N {id: 1})");

        let err = session
            .check_direct_write_allowed()
            .expect_err("expected a refusal");
        assert!(err.contains("does not support direct writes"));

        // A caller that ignores the precondition does not lose data silently:
        // the write applies, and the session refuses every later durability
        // operation until a checkpoint folds it in.
        {
            let mut graph = session.write();
            graph.bump_version();
        }
        match commit_query(&session, "CREATE (:N {id: 2})") {
            CommitOutcome::DurabilityFailed { ref error } => {
                assert!(error.contains("Session::write"), "message was: {error}")
            }
            other => panic!("expected DurabilityFailed, got {other:?}"),
        }
        assert!(session
            .sync()
            .expect_err("expected a refusal")
            .contains("Session::write"));

        // A checkpoint is the documented repair: it folds the direct write in
        // and starts a fresh log.
        session.save(&path.to_string_lossy(), true).unwrap();
        assert!(session.sync().is_ok());
        assert!(matches!(
            commit_query(&session, "CREATE (:N {id: 2})"),
            CommitOutcome::Committed { .. }
        ));
    }

    #[test]
    fn transact_latches_a_durable_session_too() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.kgl");
        let session = fresh(&path);

        session
            .transact(|working| {
                working.bump_version();
                Ok::<_, &'static str>(())
            })
            .unwrap();
        assert!(matches!(
            commit_query(&session, "CREATE (:N {id: 1})"),
            CommitOutcome::DurabilityFailed { .. }
        ));
    }

    // ── accounting ──────────────────────────────────────────────────

    #[test]
    fn a_commit_with_no_writes_appends_no_frame() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.kgl");
        let wpath = wal_path(&path);
        let session = fresh(&path);
        let header_len = std::fs::metadata(&wpath).unwrap().len();

        // (a) never materialised.
        assert!(matches!(
            session.commit(session.begin(), true),
            CommitOutcome::NoWritesNoOp
        ));
        // (b) materialised but captured nothing.
        let mut tx = session.begin();
        tx.working_mut().unwrap();
        assert!(matches!(
            session.commit(tx, true),
            CommitOutcome::Committed { .. }
        ));

        assert_eq!(
            std::fs::metadata(&wpath).unwrap().len(),
            header_len,
            "an empty commit must not append a frame"
        );
        assert_eq!(
            session.next_lsn(),
            Some(1),
            "an empty commit must not consume an LSN"
        );

        // Non-vacuity: a real write does append and does consume one.
        commit_query(&session, "CREATE (:N {id: 1})");
        assert!(std::fs::metadata(&wpath).unwrap().len() > header_len);
        assert_eq!(session.next_lsn(), Some(2));
    }

    #[test]
    fn sync_is_an_error_on_a_non_durable_session() {
        let session = Session::new(DirGraph::new());
        assert!(session.durability().is_none());
        let err = session.sync().expect_err("expected a refusal");
        assert!(err.contains("write-ahead log"), "message was: {err}");
    }
}
