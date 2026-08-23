//! Durable-open and checkpoint orchestration, independent of any binding.
//!
//! The WAL machinery itself lives elsewhere (`graph/wal.rs` for the frame
//! format and recovery scan, `graph/mutation/wal_replay.rs` for replay,
//! `graph/storage/recording.rs` for write capture). What sits here is the
//! *orchestration* — the orderings that make those pieces add up to crash
//! safety — so a durability owner is assembled the same way whichever binding
//! is doing the assembling.
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
//!    then the binding's own save, then [`checkpoint_epilogue`]) — why each
//!    step sits where it does is on those two functions. `next_lsn`
//!    deliberately keeps climbing across the truncation: restarting it would
//!    let a stale pre-checkpoint frame carry the same LSN as a fresh one,
//!    making the replay gate meaningless.
//!
//! The third ordering — *append the frame, then publish the commit* — is a
//! property of how each owner publishes writes, not something they share. See
//! [`Session::commit`](crate::graph::session::Session::commit).
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
//!
//! The same predicate is owed by openers that take no durability argument
//! ([`ensure_recovered`]) and by saves from owners that never had one
//! ([`ensure_save_target_recovered`]) — in both cases the damage runs past the
//! missing reads into a rollback of already-saved state; those two docs carry
//! the mechanism.

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
/// - **A graph whose capture wrapper a log already owns** — the observable
///   signature of another durable owner over the same data. Two owners sharing
///   one log interleave their `next_lsn` counters and each checkpoint stamps a
///   `checkpoint_lsn` the other's frames sit below, so replay silently drops
///   committed data. A wrapper installed by change data capture alone claims no
///   ownership and does not refuse.
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
    let frames = read_sidecar(&wpath)?;
    let checkpoint_lsn = graph.checkpoint_lsn;

    if !level.logs() {
        if unreplayed(&frames, checkpoint_lsn) {
            return Err(DurableOpenError::Refused(format!(
                "the write-ahead log at '{}' holds commits this checkpoint does not \
                 contain, and durability level 'off' would neither replay them nor keep \
                 them — the next checkpoint would truncate the log and the commits would \
                 be gone. Open with level 'full' or 'normal' to replay and continue the \
                 log, {DISCARD_EXIT}",
                wpath.display(),
            )));
        }
        return Ok(None);
    }

    if graph.graph.is_wal_owner() {
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
    // Replay BEFORE wrapping, or the replay's own writes enter the capture
    // buffer and the next commit logs them all over again.
    let max_lsn = apply_frames(dir, &frames, checkpoint_lsn).map_err(DurableOpenError::Replay)?;
    wrap_for_durability(dir);
    // With change data capture already enabled the graph was wrapped before the
    // replay ran, so the replayed writes sit in the capture buffer describing
    // frames this log already holds: handing them to the WAL at the next commit
    // would log every recovered write a second time. Drop them — and with them
    // the CDC events for changes no consumer of this stream saw happen live.
    // (No-op on the ordinary path, where the wrap above created the buffer.)
    if let Some(rg) = dir.graph.recording_mut() {
        let _ = rg.take_ops();
    }

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
    // Drained through the CDC seam: anything still buffered here describes
    // committed changes that the checkpoint has now folded in, so a change
    // stream must see them before they are dropped.
    crate::graph::cdc::drain_at_commit(graph);
    wal.reset()
}

/// Refuse a **log-less** open of `checkpoint_path` while its sidecar still
/// holds frames the checkpoint at `checkpoint_lsn` does not contain.
///
/// The companion to [`open_log`]'s `off` refusal, for openers that take no
/// durability argument and attach no log — [`open_or_create_graph`] and every
/// binding built on it (the MCP and Bolt servers, the CLI). They read the
/// `.kgl` alone, so without this they open a graph that is silently missing
/// committed writes and, worse, may then *save* over the path: the save
/// neither stamps `checkpoint_lsn` nor truncates the sidecar, so the stale
/// frames survive it and the next durable open replays them over the newer
/// state. That is a rollback of saved data, not merely a stale read, which is
/// why the refusal covers reads too — an opener that can later publish cannot
/// be distinguished at open time from one that only looks.
///
/// Frames at or below `checkpoint_lsn` are crash residue between the `.kgl`
/// write and the truncation, exactly as in [`open_log`], and open fine.
///
/// **Not called by [`load_file`].** That is the primitive the durable path is
/// itself built on — every durable owner loads the checkpoint and *then*
/// replays — so a guard there would make recovery unreachable. It is also the
/// documented way to read a graph another process is writing durably, where a
/// sidecar ahead of the checkpoint is the normal steady state rather than a
/// fault.
///
/// [`open_or_create_graph`]: crate::graph::io::open::open_or_create_graph
/// [`load_file`]: crate::graph::io::file::load_file
pub fn ensure_recovered(
    checkpoint_path: &Path,
    checkpoint_lsn: u64,
) -> Result<(), DurableOpenError> {
    if let Some(wpath) = unrecovered_sidecar(checkpoint_path, checkpoint_lsn)? {
        return Err(DurableOpenError::Refused(format!(
            "the write-ahead log at '{}' holds commits this checkpoint does not contain, \
             and this open attaches no log — it would neither replay them nor keep them, \
             and the first save over this path would strand them in front of a newer \
             checkpoint for a later durable open to replay back over it. Open the graph \
             through a durable entry point (a durable graph handle, or a Session, at \
             level 'full' or 'normal') to replay them first, {DISCARD_EXIT}",
            wpath.display(),
        )));
    }
    Ok(())
}

/// Refuse a **checkpoint write** to `checkpoint_path` while its sidecar holds
/// frames that a graph stamped `checkpoint_lsn` would strand in front of it.
///
/// The save-side half of [`ensure_recovered`]'s rule, for the route it cannot
/// cover: [`load_file`] is deliberately unguarded, so a non-durable owner still
/// *reads* a path whose sidecar runs ahead — and may then save back over it,
/// stranding the frames in front of the newer `.kgl` for the next durable open
/// to replay back over it. Refusing the save closes that without taking the
/// read away.
///
/// `checkpoint_lsn` is the stamp the *outgoing* `.kgl` will carry, so the
/// frames this refuses on are exactly the ones a later durable open would
/// replay over it — replay is gated on that same stamp. A durable owner's
/// checkpoint therefore passes by construction rather than by exemption: step 2
/// of the checkpoint order ([`checkpoint_prologue`]) stamps `next_lsn - 1`
/// before the save, which is at or above every frame in its own log.
///
/// [`load_file`]: crate::graph::io::file::load_file
pub fn ensure_save_target_recovered(
    checkpoint_path: &Path,
    checkpoint_lsn: u64,
) -> Result<(), DurableOpenError> {
    if let Some(wpath) = unrecovered_sidecar(checkpoint_path, checkpoint_lsn)? {
        return Err(DurableOpenError::Refused(format!(
            "the write-ahead log at '{}' holds commits the graph being saved does not \
             contain, and saving here would strand them: they sit past this checkpoint's \
             log-sequence stamp, so a later durable open would replay them back over the \
             state this save is about to write. Open the graph through a durable entry \
             point (a durable graph handle, or a Session, at level 'full' or 'normal') to \
             replay them first, {DISCARD_EXIT}",
            wpath.display(),
        )));
    }
    Ok(())
}

/// The one predicate both refusals ask, so an open and a save can never
/// disagree about what counts as unrecovered.
fn unrecovered_sidecar(
    checkpoint_path: &Path,
    checkpoint_lsn: u64,
) -> Result<Option<std::path::PathBuf>, DurableOpenError> {
    let wpath = wal_path(checkpoint_path);
    let frames = read_sidecar(&wpath)?;
    Ok(unreplayed(&frames, checkpoint_lsn).then_some(wpath))
}

/// Shared tail for every refusal: the frames are discardable only deliberately.
const DISCARD_EXIT: &str = "or move the sidecar aside first to deliberately discard those commits.";

/// Read (do not truncate) whatever the last durable owner left behind. Torn
/// tails stop the scan; see `wal::read_frames`.
fn read_sidecar(wpath: &Path) -> Result<Vec<WalFrame>, DurableOpenError> {
    recover(wpath).map_err(|e| {
        DurableOpenError::Io(format!(
            "failed to read the write-ahead log at '{}': {e}",
            wpath.display()
        ))
    })
}

fn unreplayed(frames: &[WalFrame], checkpoint_lsn: u64) -> bool {
    frames.iter().any(|f| f.lsn > checkpoint_lsn)
}

/// The `Recording(Forked)` composition — a durable owner opened while a lazy
/// view is outstanding.
///
/// [`open_log`] takes `make_dir_graph_mut` *before* it wraps, so a graph with a
/// live reader is forked first and the capture layer then wraps an **overlay**,
/// not a plain `Memory`. Every durability test until 2026-08-15 owned its graph
/// outright, so `RecordingGraph<GraphBackend>` had only ever wrapped `Memory`,
/// `Mapped` or `Disk`; the overlay composition — where every captured
/// `NodeIndex` is one the overlay handed out and every write is copy-on-write
/// against a shared base — was entirely unexercised.
///
/// The failure it guards is silent and unrecoverable: ops resolved against the
/// wrong graph, or writes that never reach the capture buffer at all, produce a
/// log that replays into a graph missing committed data, and nothing complains
/// until the crash.
#[cfg(test)]
mod recording_over_a_fork_tests {
    use super::*;
    use crate::datatypes::Value;
    use crate::graph::io::file::{load_file, save_graph};
    use crate::graph::session::execute::{execute_mut, ExecuteOptions};
    use crate::graph::storage::recording::resolve_ops;
    use crate::graph::storage::GraphRead;
    use std::collections::HashMap;

    fn run(graph: &mut DirGraph, query: &str) {
        let params = HashMap::new();
        let opts = ExecuteOptions::eager(&params);
        execute_mut(graph, query, &opts).unwrap_or_else(|e| panic!("query failed: {query}: {e}"));
    }

    /// Every node carrying an integer `id` and `age`, as `(id, age)`, sorted.
    fn people(graph: &DirGraph) -> Vec<(i64, i64)> {
        let mut out: Vec<(i64, i64)> = graph
            .graph
            .node_indices()
            .filter_map(|idx| graph.graph.node_view(idx))
            .filter_map(
                |node| match (node.id().into_owned(), node.get_property_value("age")) {
                    (Value::Int64(id), Some(Value::Int64(age))) => Some((id, age)),
                    _ => None,
                },
            )
            .collect();
        out.sort();
        out
    }

    /// A write taken on a `Recording(Forked)` backend must be logged, and the
    /// log must replay onto the checkpoint that predates it.
    ///
    /// The held view is the second half: it must see neither the write nor the
    /// replay, before or after the crash. An overlay that leaked into its
    /// shared base would show up as the writer's data appearing under the
    /// reader's snapshot — with the log still describing it, so recovery would
    /// then apply it twice.
    #[test]
    fn a_durable_write_over_a_held_view_is_logged_and_replays() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("durable.kgl");
        let path_str = path.to_string_lossy().to_string();

        let mut seed = Arc::new(DirGraph::new());
        run(
            make_dir_graph_mut(&mut seed),
            "CREATE (:Person {id: 1, name: 'Alice', age: 30})",
        );
        save_graph(&mut seed, &path_str).unwrap();

        let mut writer = load_file(&path_str).unwrap();
        // The lazy view: held across the durable open, the write, and the crash.
        let view = Arc::clone(&writer);
        let checkpoint_state = people(&view);
        assert_eq!(checkpoint_state, vec![(1, 30)], "fixture");

        let (mut wal, next_lsn) = open_log(&mut writer, &path, DurabilityLevel::Full)
            .expect("a full durable open over a clean checkpoint")
            .expect("level 'full' must hand back a log");
        {
            let recording = writer
                .graph
                .recording()
                .expect("open_log must wrap the graph for capture");
            assert!(
                recording.inner().is_forked(),
                "precondition: the held view must have forked the graph *before* the \
                 capture layer wrapped it, or this is the plain Memory arm again"
            );
        }

        let raw = {
            let dir = make_dir_graph_mut(&mut writer);
            run(dir, "MATCH (p:Person {id: 1}) SET p.age = 31");
            run(dir, "CREATE (:Person {id: 2, name: 'Bob', age: 7})");
            assert!(
                dir.graph
                    .recording()
                    .is_some_and(|rg| rg.inner().is_forked()),
                "both writes must stay overlay-expressible under the wrapper, or the \
                 composition under test flattened before it was measured"
            );
            dir.graph
                .recording_mut()
                .expect("the capture layer must survive the writes")
                .take_ops()
        };
        assert!(
            !raw.is_empty(),
            "a write through Recording(Forked) must reach the capture buffer — an empty \
             buffer here is an unlogged commit, i.e. silent data loss on the next crash"
        );
        let ops = {
            let dir = writer.as_ref();
            resolve_ops(&raw, &dir.graph, &dir.interner, |idx| {
                dir.secondary_label_names(idx)
            })
        };
        wal.append(&WalFrame { lsn: next_lsn, ops }).unwrap();
        wal.sync().unwrap();

        let live = people(&writer);
        assert_eq!(live, vec![(1, 31), (2, 7)], "the writer's own state");

        // Crash: the log holds the frame, no checkpoint was ever taken.
        drop(wal);
        drop(writer);

        let mut recovered = load_file(&path_str).unwrap();
        assert_eq!(
            people(&recovered),
            checkpoint_state,
            "non-vacuity: the checkpoint alone must NOT contain the logged write, or \
             replay has nothing to prove"
        );
        open_log(&mut recovered, &path, DurabilityLevel::Full)
            .expect("recovery must replay the frame")
            .expect("level 'full' must hand back a log");
        assert_eq!(
            people(&recovered),
            live,
            "replaying the frame a Recording(Forked) backend produced must reconstruct \
             the writer's state exactly"
        );

        assert_eq!(
            people(&view),
            checkpoint_state,
            "the held view must never have seen the durable writer's overlay"
        );
    }
}
