//! Change data capture — an opt-in, in-process stream of the changes a graph
//! publishes, addressed by stateless `(epoch, seq)` cursors.
//!
//! ## Where events come from
//!
//! From the **write-capture buffer the write-ahead log already uses**
//! ([`crate::graph::storage::recording`]), never from a second capture path.
//! `enable` installs that wrapper if the graph has none (a durable graph
//! already carries it), and every mutation that crosses the `GraphWrite` seam
//! buffers a [`RawOp`]. At a commit boundary the buffer is drained, resolved
//! against **final** state, and published as [`CdcEvent`]s.
//!
//! ## The no-phantom invariant
//!
//! **A change that was not committed must never appear in the stream.** This
//! is the property the design is arranged around, and it is why events are
//! derived at the drain rather than at the write:
//!
//! - A **failed statement** rolls its writes back and truncates the ops it
//!   buffered (`dir_graph::rollback`), so the drain never sees them.
//! - A **rolled-back transaction** drops its working copy; the fork's buffer
//!   dies with it, undrained. `RecordingGraph::Clone` starting a fork with an
//!   empty buffer is what makes that clean rather than merely likely.
//! - A **held reader** forces the writer to fork copy-on-write. The fork
//!   shares this log through its `Arc`, so the writer's commit publishes once,
//!   into the one log the reader's and writer's handles both see.
//!
//! The cost of that arrangement is that CDC has exactly the coverage the WAL
//! has: **a change is published where a durable graph would flush a frame**,
//! and a caller driving a bare `DirGraph` has to say where its commits are, by
//! calling [`drain_at_commit`] — the same obligation the durable paths already
//! carry (see `KnowledgeGraph::flush_wal` and `Session::log_working_commit`).
//! An unpublished commit is a *missing* event; there is no arrangement here
//! that can invent one.
//!
//! ## What is deliberately not here (v1)
//!
//! - **Before-images.** Events carry after-state only; a consumer that needs
//!   `{before, after}` (Neo4j's shape) must keep its own mirror. v2.
//! - **Persistence.** The log is `#[serde(skip)]` runtime state: a `.kgl` save
//!   writes none of it and a load starts a new epoch, so a cursor never
//!   silently addresses different data. See [`CdcLog`].
//! - **Disk storage mode.** Refused at `enable`; a disk graph's change
//!   boundary is the generation publish, not this buffer.

mod event;
mod log;
#[cfg(test)]
mod tests;

pub use event::{CdcChange, CdcEvent, CdcEventKind, EdgeState, NodeState};
pub use log::{CdcLog, CdcStatus, DEFAULT_CAPACITY, MAX_CAPACITY};

use crate::error::KgError;
use crate::graph::dir_graph::DirGraph;
use crate::graph::storage::mode::{live_storage_mode, StorageMode};
use crate::graph::storage::recording::RawOp;
use std::sync::{Arc, Mutex};

/// The shared handle `DirGraph` holds. `Clone` shares it — a copy-on-write
/// view, a transaction fork and the graph they came from all publish into and
/// read from one log — while `DirGraph::independent_copy` re-mints a fresh one
/// (new epoch, empty ring), exactly as it re-mints `graph_id`.
pub type CdcHandle = Arc<Mutex<CdcLog>>;

/// Lock a log handle, tolerating poisoning like every other lock in the
/// engine: a panicking publisher must not take the stream down with it.
fn lock(handle: &CdcHandle) -> std::sync::MutexGuard<'_, CdcLog> {
    handle
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Start (or reconfigure) change data capture on `graph`.
///
/// Installs the write-capture wrapper if the graph has none — without claiming
/// write-ahead-log ownership, so a durable open is still possible afterwards
/// and the durable-only duplicate-id refusal is not imposed on a graph that
/// keeps no log ([`RecordingGraph::is_wal_owner`](crate::graph::storage::recording::RecordingGraph::is_wal_owner)).
///
/// **Re-enabling an enabled log resizes it in place** and keeps the epoch, so
/// live consumer cursors survive a capacity change; a shrink evicts from the
/// front and shows up as `earliest` advancing, like any other eviction.
///
/// # Cost
///
/// Capture is not free on the write path: every mutation buffers a `RawOp`,
/// and a wrapped backend gives up the checkpoint-free-mutation fast path, so
/// an enabled graph pays the same statement-checkpoint cost a durable graph
/// always pays. Graphs with capture *off* are untouched by any of this —
/// that separation is what the perf gate protects.
///
/// # Refusals
///
/// - **Disk storage mode.** A disk graph commits by publishing an immutable
///   generation, so the `GraphWrite` buffer this stream is derived from does
///   not describe its change boundary.
/// - **A capacity of 0, or one above [`MAX_CAPACITY`].**
// `KgError` deliberately carries structured context; boxing it here would give
// this one lifecycle call a different error type from every other engine entry
// point a binding maps.
#[allow(clippy::result_large_err)]
pub fn enable(graph: &mut DirGraph, capacity: Option<usize>) -> Result<CdcStatus, KgError> {
    if live_storage_mode(graph) == StorageMode::Disk {
        return Err(KgError::Argument(
            "change data capture is not supported for storage='disk'. A disk graph \
             commits by publishing an immutable generation, so its change boundary is \
             that publish rather than the per-mutation write capture this stream is \
             derived from — enabling here would report a stream that silently missed \
             writes. Use an in-memory or mapped graph for change data capture."
                .to_string(),
        ));
    }
    let capacity = match capacity {
        None => DEFAULT_CAPACITY,
        Some(0) => {
            return Err(KgError::Argument(
                "change-data-capture capacity must be at least 1 event; a capacity of 0 \
                 would evict every event before a consumer could read it."
                    .to_string(),
            ))
        }
        Some(requested) if requested > MAX_CAPACITY => {
            return Err(KgError::Argument(format!(
                "change-data-capture capacity {requested} exceeds the maximum of \
                 {MAX_CAPACITY} events. The log is held in memory, so its bound is a \
                 memory bound; keep a retention window a consumer can actually catch up \
                 across, not an unbounded one."
            )))
        }
        Some(requested) => requested,
    };

    match &graph.cdc {
        Some(handle) => {
            let mut log = lock(handle);
            log.resize(capacity);
            Ok(log.status())
        }
        None => {
            graph.graph.wrap_for_capture();
            let log = CdcLog::new(capacity);
            let status = log.status();
            graph.cdc = Some(Arc::new(Mutex::new(log)));
            Ok(status)
        }
    }
}

/// Stop change data capture, discard the log, and hand back whether it was on.
///
/// **The capture wrapper comes off with it — unless a write-ahead log owns
/// it.** That is the whole rule, and both halves matter: capture costs a
/// buffered op per mutation and the checkpoint-free-mutation fast path, so a
/// disable that left the wrapper behind would leave a permanent tax on a graph
/// the caller believes is back to normal; and unwrapping a WAL-owned wrapper
/// would silently stop logging a durable graph, which is data loss. A durable
/// graph therefore keeps its wrapper here and simply loses its stream.
///
/// Buffered ops die with the wrapper, and that is not a lost publish: they
/// belong to a commit boundary that has not been reached, so they were never
/// publishable — and the log they would have gone to is being dropped.
pub fn disable(graph: &mut DirGraph) -> bool {
    let was_enabled = graph.cdc.take().is_some();
    graph.graph.unwrap_capture_if_unowned();
    was_enabled
}

/// This graph's log addressing state, or `None` when capture is off.
pub fn status(graph: &DirGraph) -> Option<CdcStatus> {
    graph.cdc.as_ref().map(|handle| lock(handle).status())
}

/// Read events after the cursor position `from` (exclusive), oldest first.
///
/// `None` when capture is off. The events are cloned out of the ring so the
/// lock is not held across the caller's work — B2's `db.cdc.query` builds its
/// rows from this.
pub fn read(graph: &DirGraph, from: u64, limit: Option<usize>) -> Option<Vec<CdcEvent>> {
    graph.cdc.as_ref().map(|handle| {
        lock(handle)
            .since(from, limit)
            .into_iter()
            .cloned()
            .collect()
    })
}

/// Publish one commit's drained capture buffer as CDC events.
///
/// A no-op when capture is off, and the **only** way events reach the log.
/// Call it with ops that have been drained (so no other consumer can publish
/// them again) and with `graph` in its post-commit state (so the after-state
/// resolution reads what the commit left behind) — the same two preconditions
/// [`resolve_ops`](crate::graph::storage::recording::resolve_ops) has.
pub fn publish_drained(graph: &DirGraph, raw: &[RawOp]) {
    if raw.is_empty() {
        return;
    }
    let Some(handle) = graph.cdc.as_ref() else {
        return;
    };
    let events = event::events_from_raw(raw, &graph.graph, &graph.interner, |idx| {
        graph.secondary_label_names(idx)
    });
    if events.is_empty() {
        return;
    }
    lock(handle).append(events);
}

/// Drain the capture buffer at a commit boundary, publishing what it holds,
/// and hand the raw ops back to the write-ahead log owner.
///
/// This is the drain primitive for every owner that has no fail-closed
/// requirement of its own; `Session::log_working_commit` keeps its own drain
/// because it must distinguish "nothing captured" from "the capture seam is
/// gone" and refuse the commit in the second case.
///
/// Returns an empty vector when the graph carries no capture layer at all.
pub fn drain_at_commit(graph: &mut DirGraph) -> Vec<RawOp> {
    let raw = graph
        .graph
        .recording_mut()
        .map(|recording| recording.take_ops())
        .unwrap_or_default();
    publish_drained(graph, &raw);
    raw
}
