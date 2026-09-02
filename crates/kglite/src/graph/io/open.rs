//! Shared graph open-or-create lifecycle used by server-style bindings.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
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
/// The same mandatory-lock mechanism is documented at the disk backend tests'
/// `snapshot_files` helper, which skips `.kglite.lock` for this reason.
pub struct GraphWriterLease {
    file: File,
    /// The `.lock-owner` sidecar this holder published, kept so `Drop` can
    /// stamp the release into the record it wrote. Derived from the graph
    /// path once at acquisition rather than re-derived at teardown, so the
    /// two can never name different files.
    owner: PathBuf,
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
        Self::acquire_labeled(graph_path, timeout, None)
    }

    /// [`Self::acquire_ex`], naming the holder in the published owner record.
    ///
    /// A pid alone identifies a process on *this* machine at *this* moment; an
    /// operator running four MCP clients over one graph reads the refusal
    /// somewhere else entirely, where the pid is already meaningless. `label`
    /// is what survives that trip — "Claude Desktop", "Codex" — and it is
    /// carried as a third `label=` line so a record written by a build from
    /// before this existed still parses, and one written by this build still
    /// parses in that build (the reader ignores unknown keys, and `pid=` /
    /// `since=` keep their positions).
    pub fn acquire_labeled(
        graph_path: &Path,
        timeout: Duration,
        label: Option<&str>,
    ) -> Result<Self, LeaseRefusal> {
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
                    let owner = writer_owner_path(graph_path);
                    publish_owner_record(&owner, label);
                    // Taking write ownership is the one moment that is both
                    // rare enough to afford a directory scan and safe enough
                    // to act on what it finds: a save that died mid-write
                    // leaves a full-size `<name>.tmp.<pid>.<n>` beside the
                    // graph, and nothing else in the lifecycle ever looks for
                    // one. The reaper still proves each owner is gone before
                    // unlinking, because `lock=False` writers and unrelated
                    // processes are not covered by this lease.
                    crate::graph::io::file::reap_stale_save_temps(graph_path);
                    return Ok(Self { file, owner });
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
        // Stamp the release into the record *before* unlocking, and only in
        // that order. While the flock is held no peer can have published a
        // record, so the only record this append can reach is the one this
        // holder wrote on acquisition. After the unlock a successor may
        // already have truncated and republished, and the identical append
        // would then stamp that live holder's lease as released.
        //
        // Append rather than rewrite, so the release cannot lose the `pid=` /
        // `since=` lines a contender reads, and infallible for the same
        // reason `publish_owner_record` is: the record is naming, not
        // locking. A missing sidecar is skipped rather than created — a lone
        // `released=` line names nobody.
        if let Ok(mut owner) = OpenOptions::new().append(true).open(&self.owner) {
            let _ = writeln!(owner, "released={}", record_timestamp());
        }
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
/// escapes instead of a message naming the holder, *and* the retry loop in
/// [`GraphWriterLease::acquire_ex`] never runs, so a caller's timeout (30s for
/// the CLI's eager save paths,
/// [`crate::graph::io::write_ownership::LAZY_LEASE_ACQUIRE_TIMEOUT`] for a
/// lazily-acquired one) returns instantly instead of waiting.
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
/// better error message.
fn publish_owner_record(owner_path: &Path, label: Option<&str>) {
    let mut record = format!("pid={}\nsince={}\n", std::process::id(), record_timestamp());
    if let Some(label) = label {
        // Newlines would forge extra key=value lines in a record the reader
        // parses line-wise, so a multi-line label collapses to one line rather
        // than inventing fields.
        let flattened: String = label
            .chars()
            .map(|c| if c.is_control() { ' ' } else { c })
            .collect();
        record.push_str(&format!("label={}\n", flattened.trim()));
    }
    let _ = std::fs::write(owner_path, record);
}

/// The one clock the owner record is written from. `since=` and `released=`
/// are compared against each other by whoever reads the record, so they are
/// formatted here rather than at two call sites that could drift apart.
///
/// Second-precision UTC, matching the MCP footer's `iso8601` stamp: an
/// operator reads a refusal quoting this record next to that footer, and a
/// local-time record put the same instant two hours away from it.
fn record_timestamp() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

/// Ownership details published to `<path>.lock-owner`, read back only on the
/// contention path. The bytes are *documentation*: liveness is established by
/// the failed lock acquisition that precedes every read, so a record left
/// behind by a crashed process is never mistaken for a live holder — nothing
/// reads it unless someone currently holds the lock.
///
/// Public so a binding can take the pid as data rather than parsing prose —
/// see [`LeaseRefusal`].
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct LeaseHolder {
    /// The holding process id, when the record could be read.
    pub pid: Option<u32>,
    /// RFC-3339 timestamp of when the holder took the lease. Written in UTC;
    /// records from before 0.16.21 carry a local offset, and both parse.
    pub since: Option<String>,
    /// Operator-facing name the holder published for itself, when it published
    /// one. Absent for every holder that took the lease without a label, and
    /// for records written before the field existed.
    pub label: Option<String>,
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
                Some(("label", value)) if !value.trim().is_empty() => {
                    holder.label = Some(value.trim().to_string())
                }
                // Unknown keys are skipped, which is what lets the record gain
                // fields without breaking older readers. `released=` is
                // deliberately among them: a contender only ever reads this
                // after its own lock attempt failed, so a record it can see is
                // one whose holder is still holding — the released line is
                // reachable here only in the few instructions between the
                // append and the unlock, and reporting it would say nothing
                // the flock has not already settled.
                _ => {}
            }
        }
        holder
    }

    /// Whether the reported holder is this very process. Exposed so a binding
    /// rendering its own message gets the distinction [`Self::describe`] makes
    /// without parsing its prose.
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
        match (self.label.as_deref(), self.pid, self.since.as_deref()) {
            (Some(label), Some(pid), Some(since)) => {
                format!("\"{label}\" (pid {pid}, since {since})")
            }
            (Some(label), Some(pid), None) => format!("\"{label}\" (pid {pid})"),
            (Some(label), None, _) => format!("\"{label}\""),
            (None, Some(pid), Some(since)) => format!("pid {pid} (since {since})"),
            (None, Some(pid), None) => format!("pid {pid}"),
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

/// Identity of a graph path at load/save time — the value a writer compares
/// to decide whether the path is still the one its graph was read from.
///
/// What counts as "changed" depends on the shape of the path, because each
/// shape has a different publish mechanism and a different kind of noise:
///
/// - a **regular file** is replaced atomically by every save, so its own
///   metadata (size, mtime, inode) is the signal;
/// - a **disk-graph directory with a `CURRENT` pointer** is that pointer alone
///   — its metadata plus its bytes. The root directory's own mtime and size are
///   deliberately *not* part of it: a writer mints its scratch
///   (`.working-<pid>-<nonce>/`, `.kglite.lock`) inside the root, and a server
///   that folded the root in would refuse to publish its own write. A publish
///   always replaces `CURRENT` (new inode, new bytes), so the pointer is the
///   complete change signal;
/// - a **legacy flat directory** (no pointer) has no publish signal at all — its
///   files are rewritten in place — so it is keyed on the directory's own
///   inode: it changes only if the directory is replaced wholesale, and the
///   first save under a lease migrates it to the pointer shape;
/// - a **missing path** is its own shape, equal only to another missing path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphFileIdentity {
    shape: Shape,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Shape {
    Missing,
    File(MetadataIdentity),
    Generation(MetadataIdentity, Vec<u8>),
    LegacyDir(DirIdentity),
}

/// A directory's identity without its contents' churn: which directory, not
/// what is in it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct DirIdentity {
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(windows)]
    handle: Arc<same_file::Handle>,
}

impl DirIdentity {
    fn of(metadata: &MetadataIdentity) -> Self {
        Self {
            #[cfg(unix)]
            device: metadata.device,
            #[cfg(unix)]
            inode: metadata.inode,
            #[cfg(windows)]
            handle: Arc::clone(&metadata.handle),
        }
    }
}

impl GraphFileIdentity {
    pub fn capture(path: &Path) -> io::Result<Self> {
        let (root, metadata) = match MetadataIdentity::capture(path) {
            Ok(captured) => captured,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Ok(Self {
                    shape: Shape::Missing,
                });
            }
            Err(error) => return Err(error),
        };
        if !metadata.is_dir() {
            return Ok(Self {
                shape: Shape::File(root),
            });
        }

        let current_path = path.join("CURRENT");
        let (current_identity, current_metadata) = match MetadataIdentity::capture(&current_path) {
            Ok(captured) => captured,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Ok(Self {
                    shape: Shape::LegacyDir(DirIdentity::of(&root)),
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
            shape: Shape::Generation(current_identity, bytes),
        })
    }

    /// When the served path was last published, as its filesystem reports it —
    /// the one field every server on the same path agrees on once refreshed.
    ///
    /// `None` for the two shapes that have no publish moment to report: a
    /// legacy flat directory, whose files are rewritten in place rather than
    /// replaced, and a path that is not there at all.
    pub fn modified(&self) -> Option<SystemTime> {
        match &self.shape {
            Shape::File(metadata) => Some(metadata.modified),
            // The `CURRENT` pointer's mtime, not the root's: swinging the
            // pointer is the publish, and the root's mtime also moves for a
            // writer's own scratch (see the shape doc above).
            Shape::Generation(current, _) => Some(current.modified),
            Shape::LegacyDir(_) | Shape::Missing => None,
        }
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
/// inferring it keeps that decision at the one place that knows the answer.
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

    /// A writer's own scratch — `.working-<pid>-<nonce>/`, `.kglite.lock` —
    /// lands *inside* the graph root and moves the root directory's mtime and
    /// size. The identity must not see that, or a disk server refuses to save
    /// its own write: only the `CURRENT` pointer says which generation is
    /// published.
    #[test]
    fn disk_identity_ignores_scratch_beside_current() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("CURRENT"), b"gen_00000000000000000001\n").unwrap();
        let before = GraphFileIdentity::capture(tmp.path()).unwrap();
        std::fs::create_dir(tmp.path().join(".working-1-1")).unwrap();
        std::fs::write(tmp.path().join(".kglite.lock"), b"pid=1\n").unwrap();
        let after = GraphFileIdentity::capture(tmp.path()).unwrap();
        assert_eq!(
            before, after,
            "root-directory churn is not a generation change"
        );

        // Same for a legacy flat directory, which is keyed on the directory
        // itself; its first save then migrates it to the pointer shape, and
        // that migration *is* a change.
        let legacy = tempfile::tempdir().unwrap();
        std::fs::write(legacy.path().join("metadata.json"), b"{}").unwrap();
        let before = GraphFileIdentity::capture(legacy.path()).unwrap();
        std::fs::create_dir(legacy.path().join(".working-1-1")).unwrap();
        let after = GraphFileIdentity::capture(legacy.path()).unwrap();
        assert_eq!(
            before, after,
            "scratch inside a legacy root is not a change"
        );
        std::fs::write(legacy.path().join("CURRENT"), b"gen_00000000000000000001\n").unwrap();
        let migrated = GraphFileIdentity::capture(legacy.path()).unwrap();
        assert_ne!(before, migrated, "gaining a CURRENT pointer is a change");
    }

    /// The cross-server identity: two processes serving one path disagree on
    /// every counter they keep themselves, so what they report has to come off
    /// the filesystem. Only the two atomically-republished shapes have such a
    /// moment; the other two must say so rather than invent one.
    #[test]
    fn modified_reports_a_publish_moment_only_for_the_republished_shapes() {
        let tmp = tempfile::tempdir().unwrap();

        let file = tmp.path().join("graph.kgl");
        std::fs::write(&file, b"bytes").unwrap();
        let file_modified = GraphFileIdentity::capture(&file).unwrap().modified();
        assert_eq!(
            file_modified,
            Some(std::fs::metadata(&file).unwrap().modified().unwrap()),
            "a regular file reports its own mtime"
        );

        // A generation directory reports the `CURRENT` pointer's mtime — the
        // moment the publish swung it — not the root's.
        let disk = tmp.path().join("disk");
        std::fs::create_dir(&disk).unwrap();
        let current = disk.join("CURRENT");
        std::fs::write(&current, b"gen_00000000000000000001\n").unwrap();
        assert_eq!(
            GraphFileIdentity::capture(&disk).unwrap().modified(),
            Some(std::fs::metadata(&current).unwrap().modified().unwrap()),
        );

        let legacy = tmp.path().join("legacy");
        std::fs::create_dir(&legacy).unwrap();
        std::fs::write(legacy.join("metadata.json"), b"{}").unwrap();
        assert_eq!(
            GraphFileIdentity::capture(&legacy).unwrap().modified(),
            None,
            "a legacy flat directory is rewritten in place and has no publish moment"
        );

        assert_eq!(
            GraphFileIdentity::capture(&tmp.path().join("absent"))
                .unwrap()
                .modified(),
            None,
        );
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

    /// The record is parsed line-wise by builds on both sides of the label's
    /// introduction, so `label=` is appended after the two fields those builds
    /// read positionally — `tests/test_cli_shell_smoke.py` asserts the record
    /// still starts with `pid=`.
    #[test]
    fn a_label_is_published_after_pid_and_since() {
        let tmp = tempfile::tempdir().unwrap();
        let graph = tmp.path().join("labelled.kgl");
        let _lease =
            GraphWriterLease::acquire_labeled(&graph, Duration::ZERO, Some("Claude Desktop"))
                .unwrap();

        let record = std::fs::read_to_string(writer_owner_path(&graph)).unwrap();
        let lines: Vec<&str> = record.lines().collect();
        assert!(record.starts_with("pid="), "record was {record:?}");
        assert!(lines[1].starts_with("since="));
        assert_eq!(lines[2], "label=Claude Desktop");

        let holder = LeaseHolder::read(&writer_owner_path(&graph));
        assert_eq!(holder.label.as_deref(), Some("Claude Desktop"));
        assert_eq!(holder.pid, Some(std::process::id()));
    }

    #[test]
    fn an_unlabeled_acquisition_writes_the_record_it_always_did() {
        let tmp = tempfile::tempdir().unwrap();
        let graph = tmp.path().join("plain.kgl");
        let _lease = GraphWriterLease::acquire(&graph, Duration::ZERO).unwrap();

        let record = std::fs::read_to_string(writer_owner_path(&graph)).unwrap();
        assert_eq!(record.lines().count(), 2, "record was {record:?}");
        assert!(!record.contains("label="));
        assert_eq!(
            LeaseHolder::read_once(&writer_owner_path(&graph)).label,
            None
        );
    }

    /// A record written before the field existed must keep describing its
    /// holder exactly as it used to, not degrade to "another process".
    #[test]
    fn a_record_without_a_label_line_describes_as_before() {
        let tmp = tempfile::tempdir().unwrap();
        let legacy = tmp.path().join("legacy.lock-owner");
        std::fs::write(&legacy, b"pid=999999\nsince=2026-01-01T00:00:00+01:00\n").unwrap();
        assert_eq!(
            LeaseHolder::read_once(&legacy).describe(),
            "pid 999999 (since 2026-01-01T00:00:00+01:00)"
        );
    }

    #[test]
    fn a_labeled_holder_is_described_by_its_label() {
        let holder = LeaseHolder {
            pid: Some(999999),
            since: Some("2026-01-01T00:00:00+01:00".to_string()),
            label: Some("Claude Desktop".to_string()),
        };
        assert_eq!(
            holder.describe(),
            "\"Claude Desktop\" (pid 999999, since 2026-01-01T00:00:00+01:00)"
        );
    }

    #[test]
    fn a_refusal_names_the_holders_label() {
        let tmp = tempfile::tempdir().unwrap();
        let graph = tmp.path().join("named-holder.kgl");
        let _lease =
            GraphWriterLease::acquire_labeled(&graph, Duration::ZERO, Some("Codex")).unwrap();
        let refusal = GraphWriterLease::acquire_ex(&graph, Duration::ZERO)
            .err()
            .expect("second acquire must be refused");
        assert_eq!(refusal.holder.unwrap().label.as_deref(), Some("Codex"));
    }

    /// Reads one `key=value` line out of a raw owner record, so a test can
    /// assert on a field without depending on line order.
    fn record_value(record: &str, key: &str) -> Option<String> {
        record.lines().find_map(|line| {
            line.split_once('=')
                .filter(|(name, _)| *name == key)
                .map(|(_, value)| value.trim().to_string())
        })
    }

    /// The operator complaint this answers: a record left behind by a holder
    /// that exited cleanly still read as "held since <time>", with nothing in
    /// it saying the lease was given back. Liveness was and remains the flock
    /// — the record is forensics, and it has to be honest forensics.
    #[test]
    fn a_released_lease_records_its_release() {
        let tmp = tempfile::tempdir().unwrap();
        let graph = tmp.path().join("released.kgl");
        let owner = writer_owner_path(&graph);

        let lease = GraphWriterLease::acquire(&graph, Duration::ZERO).unwrap();
        let held = std::fs::read_to_string(&owner).unwrap();
        assert!(
            !held.contains("released="),
            "a lease that is still held has not been released; record was {held:?}"
        );
        drop(lease);

        let record = std::fs::read_to_string(&owner).unwrap();
        assert!(
            record.starts_with("pid="),
            "the release must be appended, not prepended; record was {record:?}"
        );
        let since =
            record_value(&record, "since").unwrap_or_else(|| panic!("record was {record:?}"));
        let released = record_value(&record, "released")
            .unwrap_or_else(|| panic!("a released lease must say so; record was {record:?}"));
        let since = chrono::DateTime::parse_from_rfc3339(&since)
            .unwrap_or_else(|error| panic!("since={since:?} is not rfc3339: {error}"));
        let released = chrono::DateTime::parse_from_rfc3339(&released)
            .unwrap_or_else(|error| panic!("released={released:?} is not rfc3339: {error}"));
        assert!(
            released >= since,
            "a lease cannot be released before it was taken ({released} < {since})"
        );
        assert_eq!(
            record.matches("released=").count(),
            1,
            "record was {record:?}"
        );
    }

    /// The operator read `since=` beside the MCP footer's `iso8601` stamp of
    /// the same instant and saw two hours between them. Both clocks are now
    /// second-precision UTC, so the two strings are comparable by eye.
    #[test]
    fn owner_record_timestamps_are_utc() {
        let tmp = tempfile::tempdir().unwrap();
        let graph = tmp.path().join("utc.kgl");
        let owner = writer_owner_path(&graph);

        let lease = GraphWriterLease::acquire(&graph, Duration::ZERO).unwrap();
        drop(lease);

        let record = std::fs::read_to_string(&owner).unwrap();
        for key in ["since", "released"] {
            let value =
                record_value(&record, key).unwrap_or_else(|| panic!("record was {record:?}"));
            assert!(
                value.ends_with('Z'),
                "{key}={value:?} must be the UTC `Z` form the MCP footer prints"
            );
            assert!(
                !value.contains('.'),
                "{key}={value:?} must be second-precision, like the footer's iso8601"
            );
            chrono::DateTime::parse_from_rfc3339(&value)
                .unwrap_or_else(|error| panic!("{key}={value:?} is not rfc3339: {error}"));
        }
    }

    /// `Drop` is the only writer of the line, so a `SIGKILL`ed holder must
    /// leave a record that still reads as held. That asymmetry is the whole
    /// diagnostic value: a record with no `released=` is either a live holder
    /// or a crash, and the flock tells an operator which.
    #[test]
    fn a_crashed_holder_leaves_no_released_line() {
        let tmp = tempfile::tempdir().unwrap();
        let graph = tmp.path().join("crashed.kgl");
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
        let child_pid = child.id();
        let started = Instant::now();
        while !ready.exists() && started.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(ready.exists(), "child did not acquire lease");
        child.kill().unwrap();
        child.wait().unwrap();

        let record = std::fs::read_to_string(writer_owner_path(&graph)).unwrap();
        assert_eq!(
            record_value(&record, "pid"),
            Some(child_pid.to_string()),
            "record was {record:?}"
        );
        assert!(
            !record.contains("released="),
            "a killed holder never ran Drop, so nothing may claim it released \
             the lease; record was {record:?}"
        );
    }

    /// The reason the append happens *before* the unlock. Were it after, a
    /// successor could win the lock and publish its own record in the window,
    /// and the predecessor's append would then land on the new holder's
    /// record — stamping a live lease as released. Under the lock, the only
    /// record that can exist is the holder's own.
    #[test]
    fn a_successors_record_is_not_touched_by_its_predecessors_release() {
        let tmp = tempfile::tempdir().unwrap();
        let graph = tmp.path().join("succession-release.kgl");
        let owner = writer_owner_path(&graph);

        let first = GraphWriterLease::acquire(&graph, Duration::ZERO).unwrap();
        drop(first);
        let second = GraphWriterLease::acquire(&graph, Duration::ZERO).unwrap();

        let record = std::fs::read_to_string(&owner).unwrap();
        assert_eq!(
            record.matches("pid=").count(),
            1,
            "the successor truncates, so its record carries one pid; record was {record:?}"
        );
        assert!(
            !record.contains("released="),
            "the live holder's record must not inherit its predecessor's \
             release; record was {record:?}"
        );

        drop(second);
        let record = std::fs::read_to_string(&owner).unwrap();
        assert_eq!(
            record.matches("released=").count(),
            1,
            "record was {record:?}"
        );
    }

    /// The record is parsed line-wise and unknown keys are skipped, so a build
    /// from before `released=` existed reads a record carrying it exactly as
    /// it reads one without. `read`'s retry loop keys on `pid`, so a released
    /// record is neither retried longer nor returned faster than any other —
    /// and it is not treated as evidence of a live holder, because nothing
    /// reads the record unless the lock is currently held.
    #[test]
    fn a_record_with_a_released_line_still_parses_the_holder() {
        let tmp = tempfile::tempdir().unwrap();
        let held = tmp.path().join("held.lock-owner");
        let released = tmp.path().join("released.lock-owner");
        let base = "pid=999999\nsince=2026-01-01T00:00:00+01:00\nlabel=Codex\n";
        std::fs::write(&held, base).unwrap();
        std::fs::write(
            &released,
            format!("{base}released=2026-01-01T00:05:00+01:00\n"),
        )
        .unwrap();

        let held_holder = LeaseHolder::read_once(&held);
        let released_holder = LeaseHolder::read_once(&released);
        assert_eq!(released_holder.pid, Some(999999));
        assert_eq!(released_holder.label.as_deref(), Some("Codex"));
        assert_eq!(
            released_holder, held_holder,
            "an extra line must not change what the holder parses to"
        );
        assert_eq!(released_holder.describe(), held_holder.describe());

        let started = Instant::now();
        assert_eq!(LeaseHolder::read(&released), released_holder);
        assert!(
            started.elapsed() < Duration::from_millis(150),
            "a record carrying a pid returns on the first attempt; the retry \
             loop waits ~180ms and must not have run (took {:?})",
            started.elapsed()
        );
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
