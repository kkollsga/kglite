//! Online backup: a consistent single-file `.kgl` of the published graph.
//!
//! [`Session::backup`] fixes its point in time under the commit gate and both
//! session locks for the length of an `Arc` clone, then serializes that snapshot with no lock held,
//! so committers are never stalled for the serialize.
//!
//! **The point.** A commit holds the commit gate across its log append, its
//! barrier and the `Arc` swap (see [`super::durable`]). Holding the gate
//! therefore yields a published graph together with the exact LSN of the last
//! frame inside it; without it a commit between append and swap would pair the
//! graph that lacks its frame with an LSN that includes it.
//!
//! **No mutation of the snapshot.** The ordinary save prepares the graph it
//! writes in place (`prepare_kgl_write`), which on a shared `Arc` forks the
//! whole graph. The backup instead asks the read-only halves of those steps
//! whether anything would change, and only then prepares a private copy; the
//! published `Arc` is never written through. The metadata stamp is skipped
//! because it does not reach the bytes (`oldest_writer_for_save` is idempotent).
//!
//! **No sidecars.** The destination is a plain file published by temp + fsync +
//! rename + directory fsync. No `-wal` and no `.lock` is created, and no writer
//! lease is taken, so the result is one self-contained file.
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::transaction::Session;
use crate::graph::dir_graph::DirGraph;
use crate::graph::io::file::SaveError;
use crate::graph::storage::GraphRead;
use crate::graph::wal::{wal_path, DurabilityLevel};

/// Inputs to [`Session::backup`] beyond the destination.
#[derive(Debug, Clone, Default)]
pub struct BackupOptions {
    /// The checkpoint path the live graph is saved to, when the caller has one.
    /// A durable session knows it already (its log sits beside it) and derives
    /// it when this is `None`; a non-durable session cannot, so a caller that
    /// loaded the graph from a file passes that path here to have a backup over
    /// it refused.
    pub live_path: Option<PathBuf>,
}

/// What a backup wrote. `graph_version` and `lsn` are different counters:
/// the version is the session's in-memory commit count, the LSN is a position
/// in the write-ahead log.
#[derive(Debug, Clone)]
pub struct BackupReport {
    pub path: PathBuf,
    /// Size of the published file.
    pub bytes: u64,
    /// Node count of the snapshot written.
    pub nodes: usize,
    /// Relationship count of the snapshot written.
    pub relationships: usize,
    /// Version of the snapshot written.
    pub graph_version: u64,
    /// The `checkpoint_lsn` stamped into the file: the newest log frame the
    /// snapshot contains (0 when nothing was logged). `None` for a session
    /// without a write-ahead log.
    pub lsn: Option<u64>,
    /// How long both session locks were held to fix the point in time.
    pub lock_hold: Duration,
    /// Whole call, guards through the final directory fsync.
    pub elapsed: Duration,
    /// Whether the snapshot needed a private prepared copy first (column stores
    /// out of order after deletes, or stale index declarations). Costs a fork.
    pub prepared_copy: bool,
}

const DISK_REFUSAL: &str = "online backup writes a single .kgl file, which a disk-mode graph \
     cannot be: its storage is a directory of generations. Save it with save(<directory>) \
     instead, or open a memory or mapped copy to back up.";

impl Session {
    /// Write a consistent single-file backup of the published graph to `dest`.
    ///
    /// Concurrent commits keep flowing: the gate and both locks are held only to
    /// fix the point in time (which waits out a commit mid-barrier). Memory and mapped graphs are supported; a disk graph is
    /// refused. `dest` must not alias the live checkpoint (see
    /// [`BackupOptions::live_path`]) and a stray `dest-wal` holding commits the
    /// previous `dest` lacks is refused; an existing `dest` is replaced
    /// atomically.
    pub fn backup(&self, dest: &Path, opts: &BackupOptions) -> Result<BackupReport, SaveError> {
        let started = Instant::now();
        let dest_str = destination_str(dest)?;
        self.refuse_live_alias(dest, opts)?;
        crate::graph::durability::prepare_save_as_target(dest, DurabilityLevel::Off)?;

        let (snapshot, lsn, lock_hold) = self.consistent_point();
        #[cfg(test)]
        window_hook::run();
        write_backup(started, &snapshot, lsn, lock_hold, dest, dest_str)
    }

    /// Refuse a destination that is the live checkpoint, by path, by the log
    /// sidecar the durable state appends to, or by the caller-supplied path.
    fn refuse_live_alias(&self, dest: &Path, opts: &BackupOptions) -> Result<(), SaveError> {
        let wal = self
            .durable
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .map(|state| state.wal_file().to_path_buf());
        let mut aliased = false;
        // The log is `<live>-wal`, so the session knows its checkpoint even
        // when the caller supplied no `live_path`.
        let derived_live = wal.as_ref().and_then(|w| {
            let name = w.file_name()?.to_str()?.strip_suffix("-wal")?.to_owned();
            Some(w.with_file_name(name))
        });
        for live in opts.live_path.iter().chain(derived_live.iter()) {
            aliased |= crate::graph::durability::same_checkpoint_path(live, dest)
                .map_err(|e| SaveError::Io(e.to_string()))?;
            refuse_link_alias(live, dest)?;
        }
        if let Some(wal) = &wal {
            aliased |= crate::graph::io::open::same_existing_file(wal, &wal_path(dest))
                .map_err(|e| SaveError::Io(e.to_string()))?;
        }
        if aliased {
            return Err(alias_refusal(dest));
        }
        Ok(())
    }

    /// The published graph and the LSN of the last frame it contains, fixed
    /// together under the commit gate (so no commit is between its append and
    /// its swap), then graph, then durability. The second value is `None` for a
    /// non-durable session.
    fn consistent_point(&self) -> (Arc<DirGraph>, Option<u64>, Duration) {
        let _commit = self.lock_commit_gate();
        let graph = self.graph.lock().unwrap_or_else(|p| p.into_inner());
        let durable = self.durable.lock().unwrap_or_else(|p| p.into_inner());
        let held = Instant::now();
        let snapshot = Arc::clone(&graph);
        let lsn = durable.as_ref().map(|state| state.last_lsn());
        drop(durable);
        drop(graph);
        (snapshot, lsn, held.elapsed())
    }
}

/// Write a backup of a snapshot the caller already fixed, for a holder that is
/// not a [`Session`] (the Python wheel's single-owner graph). `lsn` is the
/// `checkpoint_lsn` the snapshot contains (`None` without a write-ahead log);
/// `opts.live_path` is the only alias source, since there is no session log to
/// consult. Same guards, same writer, same atomic publish as
/// [`Session::backup`]; `lock_hold` is zero because the caller held no lock.
pub fn backup_snapshot(
    snapshot: &Arc<DirGraph>,
    lsn: Option<u64>,
    dest: &Path,
    opts: &BackupOptions,
) -> Result<BackupReport, SaveError> {
    let started = Instant::now();
    let dest_str = destination_str(dest)?;
    if let Some(live) = &opts.live_path {
        if crate::graph::durability::same_checkpoint_path(live, dest)
            .map_err(|e| SaveError::Io(e.to_string()))?
        {
            return Err(alias_refusal(dest));
        }
        refuse_link_alias(live, dest)?;
    }
    crate::graph::durability::prepare_save_as_target(dest, DurabilityLevel::Off)?;
    write_backup(started, snapshot, lsn, Duration::ZERO, dest, dest_str)
}

fn destination_str(dest: &Path) -> Result<&str, SaveError> {
    dest.to_str().ok_or_else(|| {
        SaveError::Refused(format!(
            "backup destination '{}' is not valid UTF-8",
            dest.display()
        ))
    })
}

fn alias_refusal(dest: &Path) -> SaveError {
    SaveError::Refused(format!(
        "backup destination '{}' is the live graph's checkpoint; a backup is an \
         independent copy, so choose another path",
        dest.display()
    ))
}

/// Refuse a `dest` that is a symlink or hardlink to the live checkpoint.
///
/// The atomic publish would only replace the link, leaving the live file
/// intact, but a backup aimed at an alias of the live graph is a mistake the
/// caller should hear about. A symlink to some *other* file is fine.
/// (A symlinked parent directory is caught by `same_checkpoint_path`.)
fn refuse_link_alias(live: &Path, dest: &Path) -> Result<(), SaveError> {
    let io_err = |e: std::io::Error| SaveError::Io(e.to_string());
    let is_symlink = std::fs::symlink_metadata(dest)
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false);
    if is_symlink {
        if let (Ok(resolved), Ok(live_real)) = (dest.canonicalize(), live.canonicalize()) {
            if resolved == live_real {
                return Err(alias_link_refusal(dest, "a symlink to"));
            }
        }
    }
    if crate::graph::io::open::same_existing_file(live, dest).map_err(io_err)? {
        let kind = if is_symlink {
            "a symlink to"
        } else {
            "a hardlink to"
        };
        return Err(alias_link_refusal(dest, kind));
    }
    Ok(())
}

fn alias_link_refusal(dest: &Path, kind: &str) -> SaveError {
    SaveError::Refused(format!(
        "backup destination '{}' is {kind} the live graph's checkpoint; a backup is an \
         independent copy, so choose another path",
        dest.display()
    ))
}

/// Disk refusal, preparation fork if needed, serialize, publish.
fn write_backup(
    started: Instant,
    snapshot: &Arc<DirGraph>,
    lsn: Option<u64>,
    lock_hold: Duration,
    dest: &Path,
    dest_str: &str,
) -> Result<BackupReport, SaveError> {
    if snapshot.graph.is_disk() {
        return Err(SaveError::Refused(DISK_REFUSAL.to_string()));
    }
    // A backup killed mid-write leaves a full-size `<dest>.tmp.<pid>.<n>`.
    crate::graph::io::file::reap_stale_save_temps(dest);
    let (prepared, prepared_copy) = prepared_for_write(snapshot);
    let written: &DirGraph = prepared.as_ref().unwrap_or(snapshot);
    let bytes = crate::graph::io::file::write_kgl_with_stamp(written, dest_str, true, lsn)
        .map_err(|error| SaveError::Io(error.to_string()))?;
    Ok(BackupReport {
        path: dest.to_path_buf(),
        bytes,
        nodes: written.graph.node_count(),
        relationships: written.graph.edge_count(),
        graph_version: written.version(),
        lsn,
        lock_hold,
        elapsed: started.elapsed(),
        prepared_copy,
    })
}

/// Publish `snapshot` at `dest` stamped with `lsn`, preparing a private copy
/// first when writing it as it stands would differ from a normal save. The
/// published `Arc` is never written through. Returns the file's size.
pub(crate) fn write_stamped(
    snapshot: &Arc<DirGraph>,
    dest: &str,
    lsn: u64,
) -> Result<u64, SaveError> {
    let (prepared, _) = prepared_for_write(snapshot);
    let written: &DirGraph = prepared.as_ref().unwrap_or(snapshot);
    crate::graph::io::file::write_kgl_with_stamp(written, dest, true, Some(lsn))
        .map_err(|error| SaveError::Io(error.to_string()))
}

/// A private prepared copy of `snapshot` when writing it as it stands would
/// differ from what a normal save writes; `None` when it would not. The shared
/// `Arc` is never written through: the copy is a fork of it.
fn prepared_for_write(snapshot: &Arc<DirGraph>) -> (Option<Arc<DirGraph>>, bool) {
    if !snapshot.columnar_rebuild_needed() && !snapshot.index_keys_stale() {
        return (None, false);
    }
    let mut copy = Arc::clone(snapshot);
    crate::graph::io::file::prepare_kgl_write(&mut copy);
    (Some(copy), true)
}

/// Test seam for the window between fixing the point in time and serializing:
/// the only place a test can act while a backup is "in flight" without relying
/// on timing.
#[cfg(test)]
pub(super) mod window_hook {
    use std::cell::RefCell;

    type Hook = Box<dyn FnOnce()>;

    thread_local! {
        static HOOK: RefCell<Option<Hook>> = const { RefCell::new(None) };
    }

    pub(in crate::graph::session) fn set(hook: impl FnOnce() + 'static) {
        HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
    }

    pub(super) fn run() {
        if let Some(hook) = HOOK.with(|slot| slot.borrow_mut().take()) {
            hook();
        }
    }
}
