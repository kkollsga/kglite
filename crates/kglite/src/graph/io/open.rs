//! Shared graph open-or-create lifecycle used by server-style bindings.

use std::fs::{File, OpenOptions};
use std::io::{self, Read};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use fs2::FileExt;

use crate::graph::dir_graph::DirGraph;
use crate::graph::io::file::load_file;
use crate::graph::storage::mode::{
    convert_dir_graph_to_mode, live_storage_mode, new_dir_graph_in_mode, StorageMode,
};
use crate::graph::wal::DurabilityLevel;

/// How [`open_or_create_graph`] obtained the returned graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenDisposition {
    Opened,
    Created,
}

/// Graph plus the lifecycle decision made while opening it.
pub struct OpenGraphResult {
    pub graph: Arc<DirGraph>,
    pub disposition: OpenDisposition,
    /// Identity verified stable across an existing-path load, or captured
    /// immediately after creating a new path-backed graph.
    pub identity: GraphFileIdentity,
    /// The mode the graph was in *before* an explicit request converted it, and
    /// `None` when nothing was converted — which is every open through
    /// [`open_or_create_graph`], and every request that already matched.
    /// Operators are told when a server changed the mode under them; a silent
    /// conversion is as hard to notice as a silently ignored flag.
    pub converted_from: Option<StorageMode>,
}

/// Cross-process writer ownership for a graph path. Both sidecars are
/// persistent; OS lock teardown, not PID-file deletion, owns liveness.
///
/// Ownership is split across two files on purpose:
///
/// - `<path>.lock` is the lock token, and holds **no** data. It is what
///   [`fs2`] locks, and it is deliberately still this exact path so that a
///   binary from before the split still excludes, and is excluded by, this one.
/// - `<path>.lock-owner` carries the human-readable `pid` / `since` record
///   used to name the holder in an error message.
///
/// The record cannot live inside the lock file. `fs2` locks via `flock` on
/// Unix, which is *advisory* — a contender can still read the bytes — but via
/// `LockFileEx` on Windows, whose locks are **mandatory** over the whole range
/// (`fs2` passes `(0, !0, !0)`). There, an exclusive lock makes the file
/// unreadable to every other handle, including other handles in the holder's
/// own process, and a contender's read fails with `ERROR_LOCK_VIOLATION` (33)
/// rather than returning bytes. Keeping the record in an unlocked sibling is
/// what lets the holder be *named* on every platform instead of degrading to
/// an anonymous "another process" exactly where that matters most.
///
/// The same mandatory-lock mechanism is documented at the disk backend's
/// `snapshot_files` helper, which skips `.kglite.lock` for this reason.
pub struct GraphWriterLease {
    file: File,
}

/// A refused acquisition, with the holder still **structured**.
///
/// [`GraphWriterLease::acquire`] flattens this to an `io::Error` whose message
/// names the holder in prose, which is right for a human but forces a binding
/// that wants the pid — the Java wrapper's `holder()`, an operator dashboard —
/// to regex a sentence. Bindings take this instead and re-render the message
/// their own way; the prose stays available as [`Self::error`].
#[derive(Debug)]
pub struct LeaseRefusal {
    /// Who holds the lease. `Some` only on a contention refusal, and
    /// best-effort even then: the record is published just *after* the lock is
    /// taken, so a contender losing a startup race can read an empty one. Its
    /// fields are individually optional for the same reason.
    pub holder: Option<LeaseHolder>,
    /// The error [`GraphWriterLease::acquire`] would have returned — the same
    /// kind (`WouldBlock` for contention) and the same message.
    pub error: io::Error,
}

impl From<LeaseRefusal> for io::Error {
    fn from(refusal: LeaseRefusal) -> Self {
        refusal.error
    }
}

impl GraphWriterLease {
    pub fn acquire(graph_path: &Path, timeout: Duration) -> io::Result<Self> {
        Self::acquire_ex(graph_path, timeout).map_err(io::Error::from)
    }

    /// [`Self::acquire`], keeping the holder structured on refusal. One
    /// implementation, two return shapes — `acquire` is this plus a
    /// projection, so the two can never disagree about who holds what.
    pub fn acquire_ex(graph_path: &Path, timeout: Duration) -> Result<Self, LeaseRefusal> {
        let path = writer_lease_path(graph_path);
        let started = Instant::now();
        loop {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&path)
                .map_err(LeaseRefusal::io)?;
            match file.try_lock_exclusive() {
                Ok(()) => {
                    publish_owner_record(&writer_owner_path(graph_path));
                    return Ok(Self { file });
                }
                Err(error) if is_lock_contended(&error) => {
                    if started.elapsed() >= timeout {
                        // Read the record once and derive *both* outputs from
                        // it: a second read could see a different holder and
                        // hand the caller a pid its own message contradicts.
                        let holder = LeaseHolder::read(&writer_owner_path(graph_path));
                        let message = contended_message(graph_path, &path, &holder);
                        return Err(LeaseRefusal {
                            holder: Some(holder),
                            error: io::Error::new(io::ErrorKind::WouldBlock, message),
                        });
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(error) => return Err(LeaseRefusal::io(error)),
            }
        }
    }
}

impl LeaseRefusal {
    /// A refusal that is a genuine I/O failure — nobody holds anything, so
    /// there is no holder to report.
    fn io(error: io::Error) -> Self {
        Self {
            holder: None,
            error,
        }
    }
}

impl Drop for GraphWriterLease {
    fn drop(&mut self) {
        // Closing a locked descriptor normally releases its advisory lock,
        // but doing so explicitly gives every fs2 backend the same teardown
        // boundary and lets another writer acquire immediately after drop.
        let _ = FileExt::unlock(&self.file);
    }
}

fn writer_lease_path(graph_path: &Path) -> std::path::PathBuf {
    let mut lock = graph_path.as_os_str().to_os_string();
    lock.push(".lock");
    lock.into()
}

/// Whether a failed `try_lock_*` means "someone else holds it" rather than a
/// genuine I/O failure.
///
/// Deliberately **not** an `ErrorKind` comparison. `fs2` surfaces the
/// platform's native lock errno — `EWOULDBLOCK` (35) on Unix,
/// `ERROR_LOCK_VIOLATION` (33) on Windows — and only the Unix one maps to
/// [`io::ErrorKind::WouldBlock`]; the Windows one is uncategorised. A `kind()`
/// check therefore recognises contention on Unix and silently misses it on
/// Windows, where two things then go wrong at once: the raw platform error
/// escapes instead of a message naming the holder, *and* the retry loop below
/// never runs, so a caller's timeout (the CLI and MCP server both pass 30s)
/// returns instantly instead of waiting.
///
/// [`fs2::lock_contended_error`] exists precisely so callers can compare
/// portably. The `WouldBlock` arm is kept as a belt-and-braces fallback for an
/// error that carries the kind but no raw errno.
fn is_lock_contended(error: &io::Error) -> bool {
    error.raw_os_error() == fs2::lock_contended_error().raw_os_error()
        || error.kind() == io::ErrorKind::WouldBlock
}

/// Sibling of the lock file holding the holder's identity. Never locked, so it
/// stays readable on Windows — see [`GraphWriterLease`].
fn writer_owner_path(graph_path: &Path) -> std::path::PathBuf {
    let mut owner = graph_path.as_os_str().to_os_string();
    owner.push(".lock-owner");
    owner.into()
}

/// Record who now holds the lease, for the benefit of whoever is refused next.
///
/// Truncate-then-write, so a contender reading mid-update sees either nothing
/// (reported as "another process", and retried) or the complete new record —
/// never the *previous*, now-released holder's pid, which would send someone
/// chasing a process that has already exited.
///
/// Deliberately infallible: this is naming, not locking. The caller has
/// already won the lock at this point, and failing an acquisition because a
/// cosmetic sidecar could not be written would trade a working guard for a
/// better error message. A failure here costs only the pid in a message that
/// someone else may never see.
fn publish_owner_record(owner_path: &Path) {
    let record = format!(
        "pid={}\nsince={}\n",
        std::process::id(),
        chrono::Local::now().to_rfc3339()
    );
    let _ = std::fs::write(owner_path, record);
}

/// Ownership details published to `<path>.lock-owner`, read back only on the
/// contention path. The bytes are *documentation*: liveness is established by
/// the failed lock acquisition that precedes every read, so a record left
/// behind by a crashed process is never mistaken for a live holder — nothing
/// reads it unless someone currently holds the lock.
/// Public because a binding that wants the pid must not have to regex it back
/// out of [`contended_message`]'s sentence — the Java wrapper's `holder()`
/// promised "pid, and when the lease was taken" while returning the whole
/// paragraph, which is the shape this type exists to remove.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct LeaseHolder {
    /// The holding process id, when the record could be read.
    pub pid: Option<u32>,
    /// RFC-3339 local timestamp of when the holder took the lease.
    pub since: Option<String>,
}

impl LeaseHolder {
    /// Best-effort parse. Every failure mode — unreadable file, truncated
    /// mid-write by the holder, an older release's `pid`-only format —
    /// degrades to a less specific description rather than masking the real
    /// "another writer is active" error with a parse error.
    ///
    /// Retries briefly while the record is absent, because the holder takes
    /// the lock *before* it publishes its identity: two processes racing at
    /// startup (the exact case this guard exists for) otherwise reliably read
    /// the empty window and report an unnamed "another process". The cost is
    /// paid only on the error path, where a correct message beats promptness.
    fn read(owner_path: &Path) -> Self {
        const ATTEMPTS: u32 = 10;
        for attempt in 0..ATTEMPTS {
            let holder = Self::read_once(owner_path);
            if holder.pid.is_some() {
                return holder;
            }
            if attempt + 1 < ATTEMPTS {
                std::thread::sleep(Duration::from_millis(20));
            }
        }
        Self::default()
    }

    fn read_once(owner_path: &Path) -> Self {
        let Ok(text) = std::fs::read_to_string(owner_path) else {
            return Self::default();
        };
        let mut holder = Self::default();
        for line in text.lines() {
            match line.split_once('=') {
                Some(("pid", value)) => holder.pid = value.trim().parse().ok(),
                Some(("since", value)) => holder.since = Some(value.trim().to_string()),
                _ => {}
            }
        }
        holder
    }

    /// Whether the reported holder is this very process — an un-closed handle
    /// in the caller's own code, not a deployment problem. Bindings that
    /// render their own message need the same distinction
    /// [`Self::describe`] makes, without parsing its prose.
    pub fn is_self(&self) -> bool {
        self.pid == Some(std::process::id())
    }

    fn describe(&self) -> String {
        // A self-pid hit is not a deployment problem, it is an un-closed
        // handle in the caller's own code, and saying "another process" for
        // your own pid sends people hunting a process that does not exist.
        if self.is_self() {
            return format!(
                "this same process (pid {}), which has not closed an earlier open() of it",
                std::process::id()
            );
        }
        match (self.pid, self.since.as_deref()) {
            (Some(pid), Some(since)) => format!("pid {pid} (since {since})"),
            (Some(pid), None) => format!("pid {pid}"),
            _ => "another process".to_string(),
        }
    }
}

/// The message a blocked writer sees. Names the holding process, because
/// "another writer is active" leaves an operator with nothing to act on.
///
/// The closing sentence is deliberate support-burden prevention: the lock
/// file is persistent and survives a `SIGKILL`, so the natural reflex on
/// seeing one is to delete it. Deleting it does not release the lock (the OS
/// already did that when the holder died) and only removes the record of who
/// holds it, so the message says so before the reflex fires.
fn contended_message(graph_path: &Path, lock_path: &Path, holder: &LeaseHolder) -> String {
    format!(
        "{} is open for writing by {}; only one process may write a graph at a time. \
         The lock is released automatically when that process exits, even on a crash — \
         deleting {} does not release it.",
        graph_path.display(),
        holder.describe(),
        lock_path.display()
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MetadataIdentity {
    len: u64,
    modified: SystemTime,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(windows)]
    handle: Arc<same_file::Handle>,
}

impl MetadataIdentity {
    fn capture(path: &Path) -> io::Result<(Self, std::fs::Metadata)> {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        #[cfg(not(windows))]
        let metadata = std::fs::metadata(path)?;
        #[cfg(windows)]
        let (metadata, handle) = {
            let handle = Arc::new(same_file::Handle::from_path(path)?);
            let metadata = handle.as_file().metadata()?;
            (metadata, handle)
        };

        Ok((
            Self {
                len: metadata.len(),
                modified: metadata.modified()?,
                #[cfg(unix)]
                device: metadata.dev(),
                #[cfg(unix)]
                inode: metadata.ino(),
                #[cfg(windows)]
                handle,
            },
            metadata,
        ))
    }

    fn open_snapshot(&self, _path: &Path) -> io::Result<File> {
        #[cfg(windows)]
        return self.handle.as_file().try_clone();
        #[cfg(not(windows))]
        File::open(_path)
    }
}

/// Identity of a graph path at load/save time. Disk directories include the
/// published `CURRENT` pointer bytes, so a generation promotion is detected
/// even when the root directory inode itself is unchanged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphFileIdentity {
    root: Option<MetadataIdentity>,
    current: Option<(MetadataIdentity, Vec<u8>)>,
}

impl GraphFileIdentity {
    pub fn capture(path: &Path) -> io::Result<Self> {
        let (root, metadata) = match MetadataIdentity::capture(path) {
            Ok(captured) => captured,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Ok(Self {
                    root: None,
                    current: None,
                });
            }
            Err(error) => return Err(error),
        };
        if !metadata.is_dir() {
            return Ok(Self {
                root: Some(root),
                current: None,
            });
        }

        let current_path = path.join("CURRENT");
        let (current_identity, current_metadata) = match MetadataIdentity::capture(&current_path) {
            Ok(captured) => captured,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Ok(Self {
                    root: Some(root),
                    current: None,
                });
            }
            Err(error) => return Err(error),
        };
        if current_metadata.len() > 4096 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "disk graph CURRENT pointer exceeds 4096 bytes",
            ));
        }
        let mut bytes = Vec::with_capacity(current_metadata.len() as usize);
        current_identity
            .open_snapshot(&current_path)?
            .read_to_end(&mut bytes)?;
        Ok(Self {
            root: Some(root),
            current: Some((current_identity, bytes)),
        })
    }
}

/// Open an existing graph, or create an empty graph in `create_mode` when the
/// path is absent.
///
/// Existing graphs always auto-detect their persisted storage mode. Passing
/// `None` deliberately makes a missing path an error, which lets command-line
/// bindings reject typos unless the operator explicitly opts into creation.
///
/// This function makes a lifecycle decision, not a write-ownership promise.
/// A caller that may later publish to `path` must hold its own cross-process
/// writer lease across the read/modify/save interval. Read-only callers should
/// not acquire such a lease merely to open a graph.
///
/// The graph comes back with **no write-ahead log attached**, so a path whose
/// sidecar still holds unfolded frames is refused rather than opened — see
/// [`crate::graph::durability::ensure_recovered`] for why that refusal covers
/// reads as well as writes. A caller that *will* attach a log immediately
/// afterwards wants [`open_or_create_graph_in_mode`], whose `attaching_log`
/// argument stands the refusal down because that log is the recovery.
pub fn open_or_create_graph(
    path: &Path,
    create_mode: Option<StorageMode>,
) -> io::Result<OpenGraphResult> {
    open_or_create_graph_logged(path, create_mode, DurabilityLevel::Off)
}

/// [`open_or_create_graph`], plus the caller's declaration of the log it is
/// about to attach. Private because the two public entry points express the
/// same choice in the vocabulary their own callers have.
fn open_or_create_graph_logged(
    path: &Path,
    create_mode: Option<StorageMode>,
    attaching_log: DurabilityLevel,
) -> io::Result<OpenGraphResult> {
    match std::fs::metadata(path) {
        Ok(_) => {
            let before = GraphFileIdentity::capture(path)?;
            let graph = load_file(&path.to_string_lossy())?;
            let identity = GraphFileIdentity::capture(path)?;
            if identity != before {
                return Err(io::Error::new(
                    io::ErrorKind::WouldBlock,
                    format!("graph path {} changed while it was loading", path.display()),
                ));
            }
            if !attaching_log.logs() {
                unrecovered_sidecar_check(path, graph.checkpoint_lsn)?;
            }
            return Ok(OpenGraphResult {
                graph,
                disposition: OpenDisposition::Opened,
                identity,
                converted_from: None,
            });
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }

    let mode = create_mode.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "graph path '{}' does not exist and no creation storage mode was provided",
                path.display()
            ),
        )
    })?;
    // A sidecar surviving a *deleted* checkpoint is the same hazard as one
    // surviving a stale checkpoint, and worse-signposted: a fresh graph has
    // `checkpoint_lsn` 0, so every frame in it would replay over whatever this
    // caller saves here. Checked before the graph is created, so a refused
    // disk-mode open leaves no directory behind. A caller attaching a log is
    // exempt for the same reason as above — it replays the orphaned frames onto
    // the fresh graph, which is recovery from a lost checkpoint rather than a
    // rollback of a newer one.
    if !attaching_log.logs() {
        unrecovered_sidecar_check(path, 0)?;
    }
    let graph = new_dir_graph_in_mode(mode, Some(path))
        .map_err(|message| io::Error::new(io::ErrorKind::InvalidInput, message))?;
    Ok(OpenGraphResult {
        graph: Arc::new(graph),
        disposition: OpenDisposition::Created,
        identity: GraphFileIdentity::capture(path)?,
        converted_from: None,
    })
}

/// The log-less half of the recovery-on-open rule, in the error type this
/// module speaks. [`DurableOpenError::Refused`] becomes [`io::ErrorKind::InvalidData`]
/// — the path's on-disk state is not something this entry point can safely
/// open — while an unreadable sidecar stays an uncategorised I/O failure, since
/// it says nothing about the data's shape.
fn unrecovered_sidecar_check(path: &Path, checkpoint_lsn: u64) -> io::Result<()> {
    crate::graph::durability::ensure_recovered(path, checkpoint_lsn).map_err(|e| match e {
        crate::graph::durability::DurableOpenError::Io(message) => io::Error::other(message),
        other => io::Error::new(io::ErrorKind::InvalidData, other.to_string()),
    })
}

/// Open or create at `path`, honouring `requested` on **both** branches: a
/// missing path is created in it, and an existing graph that came back in a
/// different mode is converted to it.
///
/// This is the entry point for a caller who took a mode from its user — the
/// servers' `--storage`, the wheel's `storage=`. [`open_or_create_graph`] is
/// the one for a caller whose mode is only a *creation default*, like the CLI's
/// `memory`: there, converting an existing graph would silently undo the mode
/// its checkpoint recorded, which is the defect this whole seam exists to stop.
///
/// A conversion is reported through [`OpenGraphResult::converted_from`] rather
/// than performed silently, and the graph is converted in place — the returned
/// `Arc` is the only reference at this point, and the switch moves the topology
/// rather than copying it, so no second copy of the graph is ever live.
/// Requests that have no conversion (either disk direction) fail here with the
/// reason and the alternative named, instead of serving a mode nobody asked
/// for.
///
/// # `attaching_log`
///
/// The level of the write-ahead log the caller will attach to this graph
/// *immediately* after this call — [`Session::open_durable`] or a binding's
/// durable graph handle. [`DurabilityLevel::Off`] means none, and is what every
/// caller without a durability story passes.
///
/// It is asked here because recovery-on-open is unconditional
/// ([`crate::graph::durability`]): a sidecar holding frames this checkpoint does
/// not contain makes a **log-less** open a refusal, since nothing would replay
/// them and the first save over the path would strand them. A caller that
/// attaches a log at a logging level *is* the recovery — `open_durable` replays
/// those frames before the first commit — so for it the same sidecar is the
/// normal state after a crash, not a fault. Declaring the level rather than
/// inferring it keeps that decision at the one place that knows the answer: a
/// server whose `--durability` is `off` still gets the refusal, and a graph
/// created at a logging level over an orphaned sidecar replays it instead of
/// discarding it.
///
/// The caller must actually attach the log. Passing a logging level and then
/// not opening one leaves exactly the hazard the refusal exists to stop.
///
/// [`Session::open_durable`]: crate::graph::session::Session::open_durable
pub fn open_or_create_graph_in_mode(
    path: &Path,
    requested: Option<StorageMode>,
    attaching_log: DurabilityLevel,
) -> io::Result<OpenGraphResult> {
    let mut opened = open_or_create_graph_logged(path, requested, attaching_log)?;
    let Some(requested) = requested else {
        return Ok(opened);
    };
    if opened.disposition == OpenDisposition::Created {
        return Ok(opened);
    }
    let current = live_storage_mode(&opened.graph);
    if current == requested {
        return Ok(opened);
    }
    convert_dir_graph_to_mode(
        crate::graph::handle::make_dir_graph_mut(&mut opened.graph),
        requested,
    )
    .map_err(|message| io::Error::new(io::ErrorKind::InvalidInput, message))?;
    opened.converted_from = Some(current);
    Ok(opened)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatypes::Value;
    use crate::graph::storage::GraphRead;
    use crate::graph::wal::{wal_path, DurabilityLevel, MutationOp, SyncMode, Wal, WalFrame};
    use std::process::Command;

    /// Seed a checkpoint at `path` holding `Person 1` with `age` and stamped
    /// `checkpoint_lsn`, then leave the sidecar holding one frame at `lsn` that
    /// upserts the same node with `age + 1` — the residue a durable writer
    /// leaves when it dies between a commit and its next checkpoint.
    fn checkpoint_with_pending_frame(path: &Path, checkpoint_lsn: u64, age: i64, lsn: u64) {
        let mut graph = Arc::new(DirGraph::new());
        crate::graph::mutation::wal_replay::apply_frames(
            crate::graph::handle::make_dir_graph_mut(&mut graph),
            &[person_frame(1, age)],
            0,
        )
        .unwrap();
        crate::graph::handle::make_dir_graph_mut(&mut graph).checkpoint_lsn = checkpoint_lsn;
        crate::graph::io::file::save_graph(&mut graph, &path.to_string_lossy()).unwrap();

        let mut wal = Wal::open(wal_path(path), SyncMode::Barrier).unwrap();
        wal.append(&person_frame(lsn, age + 1)).unwrap();
    }

    fn person_frame(lsn: u64, age: i64) -> WalFrame {
        WalFrame {
            lsn,
            ops: vec![MutationOp::UpsertNode {
                node_type: "Person".into(),
                id: Value::Int64(1),
                title: Value::String("Alice".into()),
                properties: vec![("age".to_string(), Value::Int64(age))],
            }],
        }
    }

    fn age_of(graph: &mut Arc<DirGraph>) -> Option<Value> {
        let dir = crate::graph::handle::make_dir_graph_mut(graph);
        let idx = dir.lookup_by_id("Person", &Value::Int64(1))?;
        dir.graph
            .node_view(idx)
            .and_then(|n| n.get_field_ref("age").map(|c| c.into_owned()))
    }

    /// The end-to-end hazard this guard closes: an opener that never consults
    /// the WAL used to open such a path, write, and save — and because that
    /// save neither stamps `checkpoint_lsn` nor truncates the sidecar, the
    /// next *durable* open replayed the stale frames over the newer state.
    /// Measured before the guard: the saved `age=3` came back as `age=2`.
    ///
    /// The open is now refused, so the sequence cannot start; the committed
    /// frame is still there for a durable open to recover, which is the whole
    /// point of refusing rather than silently discarding it.
    #[test]
    fn a_stale_sidecar_cannot_be_replayed_over_a_newer_save() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("graph.kgl");
        // Checkpoint holds age=1; a durable writer committed age=2 as frame
        // lsn 1 and died before checkpointing.
        checkpoint_with_pending_frame(&path, 0, 1, 1);

        // The MCP server's opener — no WAL awareness at all.
        let refusal =
            open_or_create_graph_in_mode(&path, Some(StorageMode::Memory), DurabilityLevel::Off)
                .err()
                .expect("an open that attaches no log must not proceed over unfolded frames");
        assert_eq!(refusal.kind(), io::ErrorKind::InvalidData);
        let message = refusal.to_string();
        assert!(message.contains("graph.kgl-wal"), "{message}");
        assert!(message.contains("holds commits this checkpoint does not contain"));
        // Both exits named: replay them, or discard them deliberately.
        assert!(message.contains("'full' or 'normal'"), "{message}");
        assert!(message.contains("move the sidecar aside"), "{message}");

        // Nothing was consumed: the commit is still recoverable durably.
        let mut recovered = crate::graph::io::file::load_file(&path.to_string_lossy()).unwrap();
        crate::graph::durability::open_log(&mut recovered, &path, DurabilityLevel::Full).unwrap();
        assert_eq!(age_of(&mut recovered), Some(Value::Int64(2)));
    }

    /// The other half of the same rule: a caller that says it is about to
    /// attach a log opens the *identical* path fine, because its log is the
    /// recovery. Both directions asserted here and above, so neither an
    /// always-guard nor a never-guard implementation can pass: dropping the
    /// `attaching_log.logs()` test makes this one fail, and inverting it makes
    /// the refusal test above fail.
    #[test]
    fn a_caller_attaching_a_log_opens_the_same_stale_sidecar() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("graph.kgl");
        checkpoint_with_pending_frame(&path, 0, 1, 1);

        for level in [DurabilityLevel::Full, DurabilityLevel::Normal] {
            let opened = open_or_create_graph_in_mode(&path, Some(StorageMode::Memory), level)
                .unwrap_or_else(|e| panic!("durable={} must open to recover: {e}", level.name()));
            let mut graph = opened.graph;
            // The frames are still there, and the log this caller now attaches
            // replays them — the recovery the refusal was protecting.
            assert_eq!(age_of(&mut graph), Some(Value::Int64(1)));
            crate::graph::durability::open_log(&mut graph, &path, level).unwrap();
            assert_eq!(age_of(&mut graph), Some(Value::Int64(2)));
        }
    }

    /// A sidecar that outlived its checkpoint entirely: at `off` the creation
    /// path refuses (the frames would be stranded in front of a brand-new
    /// graph), and a caller attaching a log recovers them onto it instead.
    #[test]
    fn an_orphaned_sidecar_refuses_creation_but_replays_for_a_log() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("orphan.kgl");
        let mut wal = Wal::open(wal_path(&path), SyncMode::Barrier).unwrap();
        wal.append(&person_frame(1, 42)).unwrap();
        drop(wal);

        open_or_create_graph_in_mode(&path, Some(StorageMode::Memory), DurabilityLevel::Off)
            .err()
            .expect("creating over an orphaned sidecar strands its commits");

        let opened =
            open_or_create_graph_in_mode(&path, Some(StorageMode::Memory), DurabilityLevel::Full)
                .expect("a durable creator recovers the orphaned commits");
        assert_eq!(opened.disposition, OpenDisposition::Created);
        let mut graph = opened.graph;
        crate::graph::durability::open_log(&mut graph, &path, DurabilityLevel::Full).unwrap();
        assert_eq!(age_of(&mut graph), Some(Value::Int64(42)));
    }

    /// Crash residue — frames the checkpoint already folded in — is not
    /// grounds to refuse, exactly as at `durable='off'`. The boundary is
    /// asserted from both sides so the comparison cannot drift: `lsn ==
    /// checkpoint_lsn` is folded in (`>=` would refuse it), `lsn ==
    /// checkpoint_lsn + 1` is not (`>` on a wrong operand order, or a `<`,
    /// would let it through).
    #[test]
    fn frames_at_or_below_the_checkpoint_still_open() {
        let tmp = tempfile::tempdir().unwrap();

        let folded = tmp.path().join("folded.kgl");
        checkpoint_with_pending_frame(&folded, 7, 1, 7);
        let opened = open_or_create_graph(&folded, Some(StorageMode::Memory))
            .expect("a frame the checkpoint already contains is harmless residue");
        assert_eq!(opened.disposition, OpenDisposition::Opened);

        let ahead = tmp.path().join("ahead.kgl");
        checkpoint_with_pending_frame(&ahead, 7, 1, 8);
        assert!(
            open_or_create_graph(&ahead, Some(StorageMode::Memory)).is_err(),
            "one frame past the checkpoint is unrecovered data"
        );
    }

    /// A sidecar outliving a *deleted* checkpoint is the same hazard wearing a
    /// missing file: the fresh graph starts at `checkpoint_lsn` 0, so every
    /// frame in the sidecar would replay over whatever this caller saves.
    #[test]
    fn a_live_sidecar_beside_a_missing_checkpoint_refuses_creation() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("gone.kgl");
        let mut wal = Wal::open(wal_path(&path), SyncMode::Barrier).unwrap();
        wal.append(&person_frame(1, 2)).unwrap();

        let refusal = open_or_create_graph(&path, Some(StorageMode::Memory))
            .err()
            .expect("creating over a live sidecar must be refused");
        assert_eq!(refusal.kind(), io::ErrorKind::InvalidData);
        assert!(!path.exists(), "a refused open must leave no graph behind");
    }

    #[test]
    fn missing_path_requires_explicit_create_mode() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("missing.kgl");
        let err = open_or_create_graph(&missing, None)
            .err()
            .expect("missing path without create mode should fail");
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
        assert!(err.to_string().contains("no creation storage mode"));
    }

    #[test]
    fn creates_requested_storage_mode_when_path_is_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let memory =
            open_or_create_graph(&tmp.path().join("memory.kgl"), Some(StorageMode::Memory))
                .unwrap();
        assert_eq!(memory.disposition, OpenDisposition::Created);
        assert!(!memory.graph.graph.is_mapped());
        assert!(!memory.graph.graph.is_disk());

        let disk_path = tmp.path().join("disk");
        let disk = open_or_create_graph(&disk_path, Some(StorageMode::Disk)).unwrap();
        assert_eq!(disk.disposition, OpenDisposition::Created);
        assert!(disk.graph.graph.is_disk());
        assert!(disk_path.is_dir());
    }

    /// A no-argument open honours what the checkpoint recorded. `create_mode`
    /// is about a *missing* path, so it must not be what decides this — the
    /// file is.
    #[test]
    fn existing_graph_opens_in_the_mode_it_recorded() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("mapped.kgl");
        let mut mapped = Arc::new(
            crate::graph::storage::mode::new_dir_graph_in_mode(StorageMode::Mapped, None).unwrap(),
        );
        crate::graph::io::file::save_graph(&mut mapped, &path.to_string_lossy()).unwrap();

        let opened = open_or_create_graph(&path, None).unwrap();
        assert_eq!(opened.disposition, OpenDisposition::Opened);
        assert!(
            opened.graph.graph.is_mapped(),
            "a mapped-saved checkpoint must reopen mapped with no storage argument at all"
        );

        // …and a memory-saved one still comes back memory, whatever the
        // creation mode says.
        let memory_path = tmp.path().join("memory.kgl");
        let mut memory = Arc::new(DirGraph::new());
        crate::graph::io::file::save_graph(&mut memory, &memory_path.to_string_lossy()).unwrap();
        let opened = open_or_create_graph(&memory_path, Some(StorageMode::Mapped)).unwrap();
        assert!(!opened.graph.graph.is_mapped());
    }

    #[test]
    fn existing_graph_is_loaded_regardless_of_create_mode() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("existing.kgl");
        let mut graph = Arc::new(DirGraph::new());
        crate::graph::io::file::save_graph(&mut graph, &path.to_string_lossy()).unwrap();

        let loaded = open_or_create_graph(&path, Some(StorageMode::Disk)).unwrap();
        assert_eq!(loaded.disposition, OpenDisposition::Opened);
        assert!(!loaded.graph.graph.is_disk());
    }

    #[test]
    fn disk_identity_tracks_current_generation_content() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("CURRENT"), b"gen_00000000000000000001\n").unwrap();
        let first = GraphFileIdentity::capture(tmp.path()).unwrap();
        std::fs::write(tmp.path().join("CURRENT"), b"gen_00000000000000000002\n").unwrap();
        let second = GraphFileIdentity::capture(tmp.path()).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn writer_lease_child() {
        let Some(graph_path) = std::env::var_os("KGLITE_LEASE_CHILD_GRAPH") else {
            return;
        };
        let ready = std::env::var_os("KGLITE_LEASE_CHILD_READY").unwrap();
        let _lease = GraphWriterLease::acquire(Path::new(&graph_path), Duration::ZERO).unwrap();
        std::fs::write(ready, b"ready").unwrap();
        std::thread::sleep(Duration::from_secs(60));
    }

    #[test]
    fn crashed_process_releases_writer_lease() {
        let tmp = tempfile::tempdir().unwrap();
        let graph = tmp.path().join("graph.kgl");
        let ready = tmp.path().join("ready");
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "graph::io::open::tests::writer_lease_child",
                "--nocapture",
            ])
            .env("KGLITE_LEASE_CHILD_GRAPH", &graph)
            .env("KGLITE_LEASE_CHILD_READY", &ready)
            .spawn()
            .unwrap();
        let started = Instant::now();
        while !ready.exists() && started.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(ready.exists(), "child did not acquire lease");
        assert!(GraphWriterLease::acquire(&graph, Duration::ZERO).is_err());
        child.kill().unwrap();
        child.wait().unwrap();
        GraphWriterLease::acquire(&graph, Duration::from_secs(2)).unwrap();
    }

    #[test]
    fn writer_lease_serializes_open_create_and_publish() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("barrier.kgl");
        {
            let _lease = GraphWriterLease::acquire(&path, Duration::ZERO).unwrap();
            let mut created = open_or_create_graph(&path, Some(StorageMode::Memory)).unwrap();
            assert_eq!(created.disposition, OpenDisposition::Created);
            crate::graph::io::file::save_graph(&mut created.graph, &path.to_string_lossy())
                .unwrap();
        }
        let _lease = GraphWriterLease::acquire(&path, Duration::ZERO).unwrap();
        let opened = open_or_create_graph(&path, Some(StorageMode::Memory)).unwrap();
        assert_eq!(opened.disposition, OpenDisposition::Opened);
    }

    /// Deliberately **not** gated to Unix. Naming the holder is the feature,
    /// and a `cfg`-gated test would let it silently degrade to "another
    /// process" on Windows — the platform where a multi-process deployment
    /// tripping this is arguably most likely — while CI stayed green.
    #[test]
    fn contended_message_names_the_holding_process() {
        let tmp = tempfile::tempdir().unwrap();
        let graph = tmp.path().join("named.kgl");
        let _lease = GraphWriterLease::acquire(&graph, Duration::ZERO).unwrap();
        let error = GraphWriterLease::acquire(&graph, Duration::ZERO)
            .err()
            .expect("second acquire must be refused");
        // Carry the raw platform error into the failure text. A refusal that
        // still reads as the OS's own lock error means the contention check
        // did not classify it, and naming the errno makes that diagnosable
        // from a CI log alone instead of needing a local repro.
        let diagnosis = format!(
            "got {error:?} (raw OS error {:?}, kind {:?}). A raw platform lock \
             error here means `is_lock_contended` failed to classify it — the \
             errno differs per platform (EWOULDBLOCK on Unix, \
             ERROR_LOCK_VIOLATION on Windows), so an `ErrorKind` comparison \
             recognises only Unix.",
            error.raw_os_error(),
            error.kind()
        );
        let message = error.to_string();
        // Same-process contention is reported as such rather than as a
        // phantom "other process" carrying the caller's own pid.
        assert!(
            message.contains(&format!("pid {}", std::process::id())),
            "the refusal must name the holding process; {diagnosis}"
        );
        assert!(message.contains("named.kgl"), "{diagnosis}");
        assert!(message.contains("does not release it"), "{diagnosis}");
    }

    /// Pins the portable classification directly, so a regression is a one-line
    /// failure rather than a confusing raw-errno leak somewhere downstream.
    /// Un-gated on purpose: the whole point is that it must hold on the
    /// platform whose errno is *not* the one `ErrorKind` understands.
    #[test]
    fn contention_is_classified_from_the_platform_lock_error() {
        assert!(
            is_lock_contended(&fs2::lock_contended_error()),
            "the error fs2 returns for a contended lock must be recognised as \
             contention on every platform"
        );
        assert!(is_lock_contended(&io::Error::from(
            io::ErrorKind::WouldBlock
        )));
        // A real failure must still propagate rather than be retried as if
        // someone else merely held the lock.
        assert!(!is_lock_contended(&io::Error::from(
            io::ErrorKind::PermissionDenied
        )));
    }

    /// The misclassification also disabled waiting: an unrecognised error
    /// returned immediately instead of entering the retry loop, so the CLI's
    /// and MCP server's 30s lease timeouts silently became zero. Asserting the
    /// call actually blocks catches that independently of the message.
    #[test]
    fn a_contended_acquire_waits_for_its_timeout() {
        let tmp = tempfile::tempdir().unwrap();
        let graph = tmp.path().join("waiting.kgl");
        let _lease = GraphWriterLease::acquire(&graph, Duration::ZERO).unwrap();

        let started = Instant::now();
        let error = GraphWriterLease::acquire(&graph, Duration::from_millis(300))
            .err()
            .expect("a held lease must still be refused after the timeout");
        assert!(
            started.elapsed() >= Duration::from_millis(250),
            "acquire returned after {:?}, so it never waited — contention was \
             not recognised and the retry loop was skipped ({error})",
            started.elapsed()
        );
    }

    /// The property whose platform split broke this: the owner record must be
    /// readable *by someone other than the holder, while the lock is held*.
    ///
    /// It is asserted as a distinct test, with the raw OS error in the failure
    /// message, because the two candidate causes look identical from the
    /// outside. A record kept inside the locked file returns bytes on Unix
    /// (`flock` is advisory) and fails with `ERROR_LOCK_VIOLATION` (33) on
    /// Windows (`LockFileEx` is mandatory) — so this test distinguishes "the
    /// read failed" from "the record was empty" instead of leaving the next
    /// person to infer it from a `None`.
    #[test]
    fn owner_record_stays_readable_while_the_lease_is_held() {
        let tmp = tempfile::tempdir().unwrap();
        let graph = tmp.path().join("readable.kgl");
        let _lease = GraphWriterLease::acquire(&graph, Duration::ZERO).unwrap();

        let owner = writer_owner_path(&graph);
        let text = std::fs::read_to_string(&owner).unwrap_or_else(|error| {
            panic!(
                "the owner record must stay readable while the lease is held, but \
                 reading {} failed: {error} (raw OS error {:?}). A mandatory-lock \
                 platform reports ERROR_LOCK_VIOLATION (33) here the moment the \
                 record is moved back inside the locked file.",
                owner.display(),
                error.raw_os_error()
            )
        });
        assert!(
            text.contains(&format!("pid={}", std::process::id())),
            "{text}"
        );

        // And the lock token itself carries no data, so nothing can come to
        // depend on reading it — which is the trap that caused this.
        let lock = writer_lease_path(&graph);
        assert_eq!(
            std::fs::metadata(&lock).unwrap().len(),
            0,
            "the lock file must stay empty; its contents are unreadable to \
             contenders on mandatory-lock platforms"
        );
    }

    /// Also un-gated: the timestamp is half of what makes the message
    /// actionable (a live writer versus one forgotten since Tuesday), so it
    /// has to survive on every platform the guard ships to.
    #[test]
    fn lease_record_carries_pid_and_acquisition_time() {
        let tmp = tempfile::tempdir().unwrap();
        let graph = tmp.path().join("record.kgl");
        let _lease = GraphWriterLease::acquire(&graph, Duration::ZERO).unwrap();
        let holder = LeaseHolder::read(&writer_owner_path(&graph));
        assert_eq!(holder.pid, Some(std::process::id()));
        assert!(holder.since.is_some(), "acquisition time must be recorded");
    }

    /// A refusal must hand the caller the pid and timestamp *as data*. Every
    /// binding downstream (the Java wrapper's `holder()`, the C ABI's holder
    /// JSON) previously had to find them inside `contended_message`'s
    /// sentence, which is a parser for prose that no one owns.
    #[test]
    fn a_refusal_carries_the_holder_structured_not_only_in_prose() {
        let tmp = tempfile::tempdir().unwrap();
        let graph = tmp.path().join("structured.kgl");
        let _lease = GraphWriterLease::acquire(&graph, Duration::ZERO).unwrap();

        let refusal = GraphWriterLease::acquire_ex(&graph, Duration::ZERO)
            .err()
            .expect("a held lease must be refused");
        let holder = refusal.holder.expect("a contention refusal names a holder");
        assert_eq!(holder.pid, Some(std::process::id()));
        assert!(
            holder.since.is_some(),
            "acquisition time must be structured"
        );
        assert!(holder.is_self(), "this process is the holder");
        assert_eq!(refusal.error.kind(), io::ErrorKind::WouldBlock);
        // The prose message is still there, and still agrees with the fields.
        assert!(refusal.error.to_string().contains("this same process"));
    }

    /// `acquire` is `acquire_ex` plus a projection: the same refusal, so the
    /// two can never diverge on kind or message.
    #[test]
    fn acquire_is_acquire_ex_projected_to_its_error() {
        let tmp = tempfile::tempdir().unwrap();
        let graph = tmp.path().join("projection.kgl");
        let _lease = GraphWriterLease::acquire(&graph, Duration::ZERO).unwrap();

        let flat = GraphWriterLease::acquire(&graph, Duration::ZERO)
            .err()
            .unwrap();
        let structured = GraphWriterLease::acquire_ex(&graph, Duration::ZERO)
            .err()
            .unwrap();
        assert_eq!(flat.kind(), structured.error.kind());
        assert_eq!(flat.to_string(), structured.error.to_string());
    }

    /// A refusal that is *not* contention reports no holder — there is none,
    /// and inventing one would send an operator after a process that never
    /// existed.
    #[test]
    fn a_non_contention_refusal_reports_no_holder() {
        let tmp = tempfile::tempdir().unwrap();
        // A lock path whose parent does not exist: the sidecar cannot be
        // created, which is an I/O failure, not a held lease.
        let graph = tmp.path().join("missing-dir").join("nowhere.kgl");
        let refusal = GraphWriterLease::acquire_ex(&graph, Duration::ZERO)
            .err()
            .expect("an uncreatable lock sidecar must refuse");
        assert!(refusal.holder.is_none(), "nobody holds an unopenable lock");
        assert_ne!(refusal.error.kind(), io::ErrorKind::WouldBlock);
    }

    /// A second holder must overwrite the first's record rather than leave a
    /// dead pid to be reported as live. The record is only ever read after a
    /// failed acquisition, so it must describe whoever holds the lock *now*.
    #[test]
    fn owner_record_is_replaced_by_each_new_holder() {
        let tmp = tempfile::tempdir().unwrap();
        let graph = tmp.path().join("succession.kgl");
        let owner = writer_owner_path(&graph);

        std::fs::write(&owner, b"pid=999999\nsince=long-ago\n").unwrap();
        let _lease = GraphWriterLease::acquire(&graph, Duration::ZERO).unwrap();

        let holder = LeaseHolder::read(&owner);
        assert_eq!(
            holder.pid,
            Some(std::process::id()),
            "a stale predecessor's pid must not survive a fresh acquisition"
        );
        assert_ne!(holder.since.as_deref(), Some("long-ago"));
    }

    #[test]
    fn holder_description_degrades_without_a_readable_record() {
        // Two cases must degrade to a usable message rather than a parse
        // failure: no record at all — a holder that has locked but not yet
        // published, or a holder running a build from before the record moved
        // out of the lock file — and a record carrying only `pid=`.
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("absent.lock");
        assert_eq!(
            LeaseHolder::read_once(&missing).describe(),
            "another process"
        );

        let legacy = tmp.path().join("legacy.lock");
        std::fs::write(&legacy, b"pid=999999\n").unwrap();
        assert_eq!(LeaseHolder::read_once(&legacy).describe(), "pid 999999");
    }

    #[test]
    fn dropping_lease_does_not_delete_replacement_path() {
        let tmp = tempfile::tempdir().unwrap();
        let graph = tmp.path().join("replacement.kgl");
        let lock = writer_lease_path(&graph);
        let moved = tmp.path().join("moved.lock");
        let lease = GraphWriterLease::acquire(&graph, Duration::ZERO).unwrap();
        std::fs::rename(&lock, &moved).unwrap();
        std::fs::write(&lock, b"replacement\n").unwrap();
        drop(lease);
        assert_eq!(std::fs::read(&lock).unwrap(), b"replacement\n");
    }
}
