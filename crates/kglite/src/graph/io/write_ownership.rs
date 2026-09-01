//! Read-modify-publish ownership of one graph file.
//!
//! Every binding that serves a `.kgl` path and may later write it back has to
//! solve the same three problems: hold the cross-process writer lease for
//! exactly as long as it has unpublished changes, refuse to overwrite a file
//! some other writer replaced in the meantime, and get back to a clean state
//! when a write fails. [`WriteOwnership`] is that logic once, so a binding
//! contributes only its own vocabulary for the refusal.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use crate::graph::dir_graph::DirGraph;
use crate::graph::handle::make_dir_graph_mut_preserving_lineage;
use crate::graph::io::file::save_graph;
use crate::graph::io::open::{GraphFileIdentity, GraphWriterLease, LeaseRefusal};

/// How long a lazily-acquired writer lease waits for the current holder.
///
/// The wait a caller actually sees is this *plus* up to 200 ms: on refusal
/// [`crate::graph::io::open::LeaseHolder`] retries the owner record 10×20 ms
/// while the holder has locked but not yet published its identity, so that a
/// refusal names somebody instead of "another process".
///
/// 250 ms is a third policy beside the two already in the tree, and the reason
/// is *when* the acquisition happens. The bolt server's startup lease is
/// `Duration::ZERO` because a server that sits silently before binding a port
/// looks hung; the CLI's eager `--save` paths use 30 s because the process
/// exists only to complete that one write and waiting for a peer to finish is
/// the whole point. A lazy lease is taken inside an interactive call that is
/// about to run a query — long enough to ride out a peer's short save without
/// failing, short enough that the caller gets an answer naming the holder
/// rather than a stalled tool call.
pub const LAZY_LEASE_ACQUIRE_TIMEOUT: Duration = Duration::from_millis(250);

/// What [`WriteOwnership::begin_write`] had to do to make the graph writable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeginWrite {
    /// The lease was already held; nothing changed.
    Held,
    /// This call took the lease. A caller whose write then fails owes the
    /// lease back — see [`WriteOwnership::discard`].
    Acquired,
}

/// Outcome of [`WriteOwnership::discard`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Discarded {
    /// Whether the graph was rolled back to the pristine snapshot. `false`
    /// means no snapshot was kept, the mutations are still in the graph, and
    /// the caller must reload the file to get back to a clean state.
    pub restored: bool,
}

/// Why a write could not proceed.
#[derive(Debug)]
pub enum WriteRefusal {
    /// Another process holds the writer lease. The holder is still structured.
    Contended(LeaseRefusal),
    /// The file is not the one this graph was loaded from any more. Deliberate
    /// dead end: this type never reloads, because a reload would drop whatever
    /// the caller re-applies on top of a fresh graph (bound embedders, an
    /// ontology) and, for a caller holding a lock across the call, deadlock on
    /// its own open path. The caller refreshes through that path and reports
    /// back with [`WriteOwnership::resynced`].
    Stale { path: PathBuf },
    /// Neither contention nor staleness: the lock file, the metadata read, or
    /// the save itself failed. A [`crate::graph::io::file::SaveError`] arrives
    /// here as the error's source, so a binding that distinguishes "refused
    /// before touching the path" can still recover it by downcast.
    Io(io::Error),
}

impl std::fmt::Display for WriteRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Contended(refusal) => write!(f, "{}", refusal.error),
            Self::Stale { path } => {
                write!(f, "{} changed on disk since it was loaded", path.display())
            }
            Self::Io(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for WriteRefusal {}

impl WriteRefusal {
    /// A refusal carries a holder only when the lock was actually contended;
    /// [`LeaseRefusal`] also carries plain I/O failures, which are nobody's
    /// contention and must not be reported as somebody's.
    fn from_lease(refusal: LeaseRefusal) -> Self {
        match refusal.holder {
            Some(_) => Self::Contended(refusal),
            None => Self::Io(refusal.error),
        }
    }
}

/// Write ownership of `path` on behalf of one in-memory graph.
///
/// Constructed with a path, always: [`GraphFileIdentity::capture`] on a
/// missing path is an all-`None` value, so a pathless graph carrying one would
/// compare equal to a graph whose file had been deleted. A graph with no file
/// has no `WriteOwnership` instead of an empty one.
pub struct WriteOwnership {
    path: PathBuf,
    lease: Option<GraphWriterLease>,
    /// Identity of `path` as of the last load or publish. The comparison that
    /// makes a lost update impossible is against this, never against a
    /// freshly-captured value.
    synced: GraphFileIdentity,
    /// `graph.version()` when the graph last matched the file.
    baseline: u64,
    /// The highest version this graph's lineage has ever reached. A rollback
    /// must land *above* it: the Cypher plan cache is keyed `(graph_id,
    /// version)` and `graph_id` survives the clone, so re-using a version the
    /// discarded lineage already cached against serves its plans.
    high_water: u64,
    pristine: Option<Arc<DirGraph>>,
    label: Option<String>,
    keep_pristine: bool,
    /// Set when the caller handed in a lease it owns the lifetime of, so
    /// [`Self::publish`] leaves it in place instead of releasing it.
    pinned_lease: bool,
}

impl WriteOwnership {
    pub fn new(
        path: PathBuf,
        identity: GraphFileIdentity,
        graph: &DirGraph,
        label: Option<String>,
        keep_pristine: bool,
    ) -> Self {
        let version = graph.version();
        Self {
            path,
            lease: None,
            synced: identity,
            baseline: version,
            high_water: version,
            pristine: None,
            label,
            keep_pristine,
            pinned_lease: false,
        }
    }

    /// Take over a lease the caller acquired itself, typically before the
    /// graph was even opened.
    ///
    /// This is how a caller keeps an acquisition ordering, or a timeout, that
    /// the lazy path does not offer: a path that is about to be *created* must
    /// be locked before the create, and the CLI's `--save` / `--save-on-exit`
    /// paths lock before loading, at 30 s, so two of them writing the same
    /// graph serialize instead of the second loading a snapshot the first is
    /// about to invalidate. `pinned` says who owns the lease's lifetime: `true`
    /// keeps it across [`Self::publish`] for a caller that exists only to
    /// complete its writes (the CLI); `false` hands it to the lazy lifecycle,
    /// released by the first publish or discard (a server that created a file
    /// and should stop excluding peers once it is on disk).
    pub fn adopt_lease(&mut self, lease: GraphWriterLease, graph: &Arc<DirGraph>, pinned: bool) {
        self.lease = Some(lease);
        self.pinned_lease = pinned;
        if self.keep_pristine && self.pristine.is_none() {
            self.pristine = Some(Arc::clone(graph));
        }
        self.note_version(graph);
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// The identity the next write is checked against.
    pub fn synced(&self) -> &GraphFileIdentity {
        &self.synced
    }

    pub fn holds_lease(&self) -> bool {
        self.lease.is_some()
    }

    /// Whether `graph` carries changes the file does not. `version` is bumped
    /// by every mutation path and by no read path, so this is exact rather
    /// than a heuristic.
    pub fn is_dirty(&self, graph: &DirGraph) -> bool {
        graph.version() != self.baseline
    }

    /// Record that the lineage reached `graph`'s version, so a later rollback
    /// clears it.
    pub fn note_version(&mut self, graph: &DirGraph) {
        self.high_water = self.high_water.max(graph.version());
    }

    /// Make the graph writable: hold the lease, and prove the file is still
    /// the one this graph came from.
    pub fn begin_write(&mut self, graph: &mut Arc<DirGraph>) -> Result<BeginWrite, WriteRefusal> {
        if self.lease.is_some() {
            self.note_version(graph);
            return Ok(BeginWrite::Held);
        }
        let lease = GraphWriterLease::acquire_labeled(
            &self.path,
            LAZY_LEASE_ACQUIRE_TIMEOUT,
            self.label.as_deref(),
        )
        .map_err(WriteRefusal::from_lease)?;
        let identity = GraphFileIdentity::capture(&self.path).map_err(WriteRefusal::Io)?;
        if identity != self.synced {
            // Released before returning: nothing has been mutated yet, so
            // holding the lease over a refusal would block every peer for a
            // window the caller cannot end.
            drop(lease);
            return Err(WriteRefusal::Stale {
                path: self.path.clone(),
            });
        }
        if self.keep_pristine {
            self.pristine = Some(Arc::clone(graph));
        }
        self.lease = Some(lease);
        self.note_version(graph);
        Ok(BeginWrite::Acquired)
    }

    /// Write the graph back to its file.
    ///
    /// Acquires the lease if it is not already held, because a *clean* graph
    /// still publishes: a server that materialized an ontology at boot has
    /// changed nothing the version counter can see, and its save is the only
    /// way that materialization reaches disk.
    pub fn publish(&mut self, graph: &mut Arc<DirGraph>) -> Result<(), WriteRefusal> {
        if self.lease.is_none() {
            let lease = GraphWriterLease::acquire_labeled(
                &self.path,
                LAZY_LEASE_ACQUIRE_TIMEOUT,
                self.label.as_deref(),
            )
            .map_err(WriteRefusal::from_lease)?;
            self.lease = Some(lease);
        }
        let identity = GraphFileIdentity::capture(&self.path).map_err(WriteRefusal::Io)?;
        if identity != self.synced {
            // A dirty graph keeps both its mutations and the lease so the
            // caller chooses — save somewhere else, or discard. A clean one
            // has nothing to lose, and a lease held over nothing would block
            // every peer for a window the caller cannot end.
            if !self.is_dirty(graph) && !self.pinned_lease {
                self.lease = None;
            }
            return Err(WriteRefusal::Stale {
                path: self.path.clone(),
            });
        }
        save_graph(graph, &self.path.to_string_lossy())
            .map_err(|error| WriteRefusal::Io(io::Error::other(error)))?;
        self.synced = GraphFileIdentity::capture(&self.path).map_err(WriteRefusal::Io)?;
        self.baseline = graph.version();
        self.note_version(graph);
        // Dropped only now, so a save that fails still has a memory-only
        // rollback. It costs the save nothing: the first mutation already
        // forked the live graph away from this snapshot, so `prepare_save`
        // sees a unique `Arc` — pinned by the MCP crate's
        // `save_does_not_deep_copy_the_active_graph`.
        self.pristine = None;
        if !self.pinned_lease {
            self.lease = None;
        }
        Ok(())
    }

    /// Roll the graph back to the last published state and release the lease.
    ///
    /// The rollback lands one version *above* everything the discarded lineage
    /// reached, not back at the baseline, so no plan cached against a
    /// mutation that never happened can be served afterwards.
    pub fn discard(&mut self, graph: &mut Arc<DirGraph>) -> Discarded {
        self.note_version(graph);
        let restored = match self.pristine.take() {
            Some(pristine) => {
                *graph = pristine;
                let cleared = self.high_water.saturating_add(1);
                make_dir_graph_mut_preserving_lineage(graph).set_version(cleared);
                self.baseline = graph.version();
                self.high_water = self.baseline;
                true
            }
            // Nothing to roll back to: the mutations stay, the graph stays
            // dirty, and the caller reloads. Deliberately not marked clean —
            // a baseline moved to the mutated version would claim the file
            // holds changes it does not.
            None => false,
        };
        self.lease = None;
        self.pinned_lease = false;
        Discarded { restored }
    }

    /// Adopt a graph the caller reloaded through its own open path.
    ///
    /// The lease is left exactly as it is: clean-and-unleased is the normal
    /// case, and a caller that reloaded while holding one is entitled to keep
    /// writing.
    pub fn resynced(&mut self, identity: GraphFileIdentity, graph: &DirGraph) {
        self.synced = identity;
        self.baseline = graph.version();
        self.pristine = None;
        self.note_version(graph);
    }

    /// Point this ownership at a different file (`save_as`). The old file's
    /// lease is released — this graph is not going back there.
    pub fn retarget(&mut self, new_path: PathBuf, identity: GraphFileIdentity) {
        self.lease = None;
        self.pinned_lease = false;
        self.path = new_path;
        self.synced = identity;
    }
}

#[cfg(test)]
#[path = "write_ownership_tests.rs"]
mod write_ownership_tests;
