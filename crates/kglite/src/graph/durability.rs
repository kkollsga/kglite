//! Durable-open and checkpoint orchestration, independent of any binding.
//!
//! The engine has shipped the WAL machinery itself since 0.14 (`graph/wal.rs`
//! for the frame format and recovery scan, `graph/mutation/wal_replay.rs` for
//! replay, `graph/storage/recording.rs` for write capture). What sits here is
//! the *orchestration* around it — the orderings that make those pieces add up
//! to crash safety — so that a durability owner is assembled the same way
//! whichever binding is doing the assembling.
//!
//! Two owners exist today and neither can be expressed in terms of the other:
//! [`Session`](crate::graph::session::Session) holds its log beside a
//! `Mutex<Arc<DirGraph>>` and appends between the OCC check and the `Arc` swap,
//! while the Python wheel's `KnowledgeGraph` holds it beside a graph it mutates
//! in place and appends after the fact. What they share is exactly what is
//! here: how a log is opened over a checkpoint, and what a checkpoint does to
//! the log on the way past.
//!
//! ## The orderings that are correctness, not preference
//!
//! 1. **Open: recover → replay → wrap → open-for-append** ([`open_log`]).
//!    Replay must happen *before* the backend is wrapped for capture, or the
//!    replay's own `GraphWrite` calls land in the capture buffer and get logged
//!    a second time. Replay is gated on the loaded graph's `checkpoint_lsn`, so
//!    frames already folded into the `.kgl` are skipped rather than rolled back
//!    over newer state.
//!
//! 2. **Checkpoint: sync → stamp → save → reset** ([`checkpoint_prologue`],
//!    then the binding's own save, then [`checkpoint_epilogue`]). The flush is
//!    load-bearing under [`DurabilityLevel::Normal`], where the tail may still
//!    be in the page cache; the stamp records how far the checkpoint consumed
//!    the log; the truncation comes last so the `.kgl` is known to have landed
//!    before the log describing the same commits is destroyed. `next_lsn`
//!    deliberately keeps climbing across the truncation: restarting it would
//!    let a stale pre-checkpoint frame carry the same LSN as a fresh one,
//!    making the replay gate meaningless.
//!
//! The third ordering — *append the frame, then publish the commit* — is not
//! here, because it is not shared: it is a property of how each owner publishes
//! writes. See [`Session::commit`](crate::graph::session::Session::commit).
//!
//! ## Recovery on open is unconditional
//!
//! Opening a path is a decision about that path's *data*, not only about how
//! future writes will be logged. A sidecar carrying frames the loaded
//! checkpoint does not contain is unrecovered data at every level, so
//! [`DurabilityLevel::Off`] is refused over one rather than silently returning
//! a graph that is missing committed writes — the first checkpoint afterwards
//! would truncate the log and destroy them for good. Frames at or below
//! `checkpoint_lsn` are the harmless residue of a crash between the `.kgl`
//! write and the truncation, and are not grounds to refuse.

use std::io;
use std::path::Path;
use std::sync::Arc;

use crate::graph::dir_graph::DirGraph;
use crate::graph::handle::make_dir_graph_mut;
use crate::graph::mutation::wal_replay::apply_frames;
use crate::graph::storage::recording::wrap_for_durability;
use crate::graph::wal::{recover, wal_path, DurabilityLevel, Wal, WalFrame};

/// Why [`open_log`] could not hand back a durability owner.
///
/// Three categories rather than one string because bindings map them to
/// different error classes — the Python wheel raises `IOError` for
/// [`Io`](Self::Io), `RuntimeError` for [`Replay`](Self::Replay) and
/// `ValueError` for [`Refused`](Self::Refused), and flattening them would
/// change what a caller's `except` clause catches.
#[derive(Debug)]
pub enum DurableOpenError {
    /// The sidecar could not be read, or could not be opened for append.
    Io(String),
    /// The recovered frames could not be applied to the checkpoint.
    Replay(String),
    /// The open is structurally unsafe and was refused: unreplayed frames at
    /// level `off`, or a graph another durability owner already holds.
    Refused(String),
}

impl std::fmt::Display for DurableOpenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(m) | Self::Replay(m) | Self::Refused(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for DurableOpenError {}

/// Open the write-ahead log for `graph`, checkpointed at `checkpoint_path`,
/// performing the full recover → replay → wrap → open-for-append ordering.
///
/// `checkpoint_path` is the `.kgl` path, not the log path; the sidecar is
/// derived from it by [`wal_path`] exactly as every binding derives it, so a
/// graph saved by one and reopened by another finds the same log.
///
/// Returns the open log plus the LSN its next frame should carry, or `None` at
/// [`DurabilityLevel::Off`] — which still *reads* the sidecar, because
/// recovery on open is unconditional (module docs).
///
/// # Refusals
///
/// - **Unreplayed frames at level `off`** — the data-loss case the module docs
///   describe.
/// - **A graph already wrapped for capture** — the observable signature of
///   another durable owner over the same data. Two owners sharing one log
///   interleave their `next_lsn` counters and each checkpoint stamps a
///   `checkpoint_lsn` the other's frames sit below, so replay silently drops
///   committed data.
///
/// **Disk-mode graphs are the caller's refusal, not this function's.** A disk
/// graph commits by publishing an immutable generation, so it keeps no logical
/// log at any level — but `off` on a disk graph is legal, and the message a
/// binding owes its users names that binding's own save API. Check
/// `graph.graph.is_disk()` before calling at a logging level.
pub fn open_log(
    graph: &mut Arc<DirGraph>,
    checkpoint_path: &Path,
    level: DurabilityLevel,
) -> Result<Option<(Wal, u64)>, DurableOpenError> {
    let wpath = wal_path(checkpoint_path);
    // Read (do not truncate) whatever the last session left behind. Torn tails
    // stop the scan; see `wal::read_frames`.
    let frames = recover(&wpath).map_err(|e| {
        DurableOpenError::Io(format!(
            "failed to read the write-ahead log at '{}': {e}",
            wpath.display()
        ))
    })?;
    let checkpoint_lsn = graph.checkpoint_lsn;

    if !level.logs() {
        if unreplayed(&frames, checkpoint_lsn) {
            return Err(DurableOpenError::Refused(format!(
                "the write-ahead log at '{}' holds commits this checkpoint does not \
                 contain, and durability level 'off' would neither replay them nor keep \
                 them — the next checkpoint would truncate the log and the commits would \
                 be gone. Open with level 'full' or 'normal' to replay and continue the \
                 log, or move the sidecar aside first to deliberately discard those \
                 commits.",
                wpath.display(),
            )));
        }
        return Ok(None);
    }

    if graph.graph.is_recording() {
        return Err(DurableOpenError::Refused(
            "this graph is already wrapped for durable capture, which means another \
             durable owner (a durable graph handle, or a Session) holds it. Two owners \
             over one write-ahead log interleave their log-sequence numbers and each \
             checkpoint invalidates the other's replay gate, so the second open is \
             refused. Take a non-durable snapshot for reads, or hand ownership over \
             instead of duplicating it."
                .to_string(),
        ));
    }

    let sync = level
        .sync_mode()
        .expect("level.logs() is true, so sync_mode is Some");

    let dir = make_dir_graph_mut(graph);
    // Replay BEFORE wrapping: the replay's own writes must not enter the
    // capture buffer, or the next commit would log them all over again.
    let max_lsn = apply_frames(dir, &frames, checkpoint_lsn).map_err(DurableOpenError::Replay)?;
    wrap_for_durability(dir);

    let wal = Wal::open(wpath.clone(), sync).map_err(|e| {
        DurableOpenError::Io(format!(
            "failed to open the write-ahead log at '{}': {e}",
            wpath.display()
        ))
    })?;
    Ok(Some((wal, max_lsn + 1)))
}

/// Checkpoint steps 1–2: flush the log, then stamp how far this checkpoint
/// will have consumed it into the graph that is about to be serialized.
///
/// The flush is load-bearing under [`DurabilityLevel::Normal`], not
/// belt-and-braces. The checkpoint truncates the log, and replay folds whatever
/// frames survive into net per-entity state. Under `full` every frame is
/// already on disk here, so a crash in the write→truncate window leaves the
/// *complete* frame set and replaying it reproduces exactly the checkpointed
/// state. Under `normal` the tail may still be in the page cache, so the same
/// crash can leave a *prefix*, and folding that prefix over a newer checkpoint
/// rolls properties back to their values as of an earlier commit — destroying
/// data that was already durably saved.
///
/// `next_lsn` is the LSN the *next* frame will carry, so the newest frame
/// folded into this snapshot is `next_lsn - 1`. An owner that has logged
/// nothing leaves the stamp at 0, i.e. replay everything, which is the ungated
/// behaviour.
pub fn checkpoint_prologue(wal: &mut Wal, next_lsn: u64, graph: &mut DirGraph) -> io::Result<()> {
    wal.sync()?;
    graph.checkpoint_lsn = next_lsn.saturating_sub(1);
    Ok(())
}

/// Checkpoint step 4: the `.kgl` now holds the full current state, so drop the
/// capture buffer (those ops are folded in) and truncate the log.
///
/// Ordered after the save on purpose: the checkpoint is known to have landed
/// before the log that describes the same commits is destroyed, and replay is
/// idempotent, so a crash between the two costs only a harmless re-apply on the
/// next open.
pub fn checkpoint_epilogue(wal: &mut Wal, graph: &mut DirGraph) -> io::Result<()> {
    if let Some(rg) = graph.graph.recording_mut() {
        let _ = rg.take_ops();
    }
    wal.reset()
}

/// Whether `frames` hold anything the checkpoint at `checkpoint_lsn` has not
/// already folded in.
fn unreplayed(frames: &[WalFrame], checkpoint_lsn: u64) -> bool {
    frames.iter().any(|f| f.lsn > checkpoint_lsn)
}
