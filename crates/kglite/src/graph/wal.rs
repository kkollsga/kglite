//! Write-ahead log for durable in-memory graphs (Stage 1 of the
//! embedded-Cypher-DB durability work).
//!
//! ## What this is
//!
//! A `.kgl-wal` sidecar holds an append-only sequence of **logical**
//! mutation frames. Each committed mutation operation appends one
//! [`WalFrame`] — a batch of [`MutationOp`]s tagged with a log-sequence
//! number (LSN) — and `fsync`s. On open, the engine loads the `.kgl`
//! checkpoint snapshot, then replays every WAL frame with
//! `lsn > DirGraph::checkpoint_lsn` to recover work committed since the
//! last checkpoint. A checkpoint (a full `.kgl` save) truncates the WAL
//! and stamps the LSN it consumed up to into the `.kgl`.
//!
//! The LSN is a **counter owned by the log**, not the graph `version`:
//! the writing binding hands out `next_lsn` and increments it (see
//! `KnowledgeGraph::flush_wal`). It is monotonic for the life of the log
//! and survives checkpoint truncation, which is what lets the stamped
//! `checkpoint_lsn` distinguish a frame the snapshot already contains
//! from one committed after it. Graph `version` is a different quantity —
//! it advances on work that is never logged — and is not a log position.
//!
//! This module owns only the **on-disk format**: the op schema, the
//! frame envelope, and crash-safe read/write. Capture (translating
//! `GraphWrite` calls into ops) and replay (applying ops to a
//! `DirGraph`) live in their own modules — kept separate so the format
//! can be tested in isolation.
//!
//! ## Logical, identity-keyed ops
//!
//! Ops are keyed by **stable logical identity**, never by petgraph
//! `NodeIndex`/`EdgeIndex` (which do not survive checkpoint load or
//! compaction). A node is `(node_type, id)`; an edge is
//! `(conn_type, src, tgt)`. Both are unique in kglite's model, so the
//! two state-changing shapes are an idempotent **upsert** (add-or-replace
//! the full property set) and a **remove**. Idempotence means replaying a
//! frame twice is harmless — important for crash recovery, where the last
//! frame before a crash may or may not have been applied to the snapshot.
//!
//! ## Crash safety of the format
//!
//! A frame is `[len: u32 LE][crc32: u32 LE][payload: codec(WalFrame)]`,
//! emitted by a **single** `write_all` (see [`append_frame`]).
//! The v2/v3 file headers select Postcard for every frame. Older headers
//! are rejected before any payload or torn-tail handling.
//! A crash mid-append leaves a torn trailing frame; [`read_frames`] stops
//! at the first short read or CRC mismatch and returns every frame up to
//! it. A torn frame is therefore *discarded*, never half-applied — the
//! atomic unit of durability is the whole frame.
//!
//! Torn-tail handling does **not** depend on `fsync`. `fsync` controls
//! *when* bytes reach stable storage, not whether a write is atomic, so a
//! torn frame has always been possible and has always been discarded. That
//! is what lets the barrier become a per-level choice below without
//! touching recovery.
//!
//! ## Durability levels
//!
//! [`DurabilityLevel`] names what a committed mutation survives; the WAL
//! itself only cares about the derived [`SyncMode`]:
//!
//! - [`DurabilityLevel::Full`] → [`SyncMode::Barrier`]. Every frame is
//!   flushed to stable storage before [`Wal::append`] returns, so an
//!   acknowledged commit survives **power loss**.
//! - [`DurabilityLevel::Normal`] → [`SyncMode::PageCache`]. The frame is
//!   handed to the OS with `write(2)` and no barrier. The page cache
//!   belongs to the kernel, not the process, so an acknowledged commit
//!   survives the **process** dying (`SIGKILL`, panic, OOM-kill) but not an
//!   OS crash or power loss.
//! - [`DurabilityLevel::Off`] → no WAL at all; durability is whatever the
//!   caller's `save()` checkpoints provide.
//!
//! Under `Normal` an OS crash can lose an arbitrary suffix of the log, but
//! never a *hole*: [`read_frames`] stops at the first frame it cannot
//! verify, so recovery always yields a **prefix**. Frames are per-commit
//! and replay is idempotent, so a prefix is a valid earlier state rather
//! than a corrupt one.
//!
//! One invariant this places on the *caller*: a checkpoint must not
//! truncate frames that are still only in the page cache, or replaying the
//! surviving prefix could revert data the checkpoint already holds. Call
//! [`Wal::sync`] before checkpointing — see its docs.

use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::datatypes::Value;

/// File magic for a kglite WAL sidecar: `KWAL`.
pub const WAL_MAGIC: [u8; 4] = *b"KWAL";

/// On-disk WAL format version *written* by this build. Bumped when the
/// frame payload gains anything an older build could not parse; the WAL is
/// a within-version recovery artefact (truncated at every checkpoint), not
/// a long-term archival format like `.kgl`.
///
/// **v2 → v3** appended [`MutationOp::SetNodeLabels`] to the op enum.
/// Postcard tags enum variants by index, so every v2 op (tags 0–3) encodes
/// byte-identically under v3 — a v2 WAL is a *strict subset* of v3 and is
/// read exactly, without a compat mirror of the old schema (see
/// [`MIN_READABLE_WAL_FORMAT_VERSION`]). The version byte still moves,
/// because the reverse direction is not safe: a v3 WAL handed to a
/// v2-writing build would hit an unknown tag, and that build's recovery
/// treats an unparseable payload as a torn tail — it would *silently
/// discard* committed frames. The header bump converts that silent data
/// loss into the loud "unsupported WAL format version" refusal such a
/// build already implements.
pub const WAL_FORMAT_VERSION: u8 = 3;

/// Oldest WAL format this build can replay. Frames from any version in
/// `MIN_READABLE_WAL_FORMAT_VERSION..=WAL_FORMAT_VERSION` decode with the
/// current [`MutationOp`] schema; see [`WAL_FORMAT_VERSION`] for why that
/// is sound rather than a shim. Reading these is deliberate
/// format-lifecycle handling: a WAL that outlived the build that wrote it
/// is exactly the crash-recovery case durability exists for, so an
/// upgraded binary must recover it, not discard it.
pub const MIN_READABLE_WAL_FORMAT_VERSION: u8 = 2;

const MAX_WAL_FRAME_BYTES: u64 = u32::MAX as u64;

/// What a committed mutation is guaranteed to survive — the durability
/// vocabulary a binding exposes to its users. Deliberately mirrors SQLite's
/// `synchronous` levels (`FULL` / `NORMAL` / `OFF`), because the audience for
/// an embedded database already knows that vocabulary and the guarantees line
/// up.
///
/// The levels are stated in terms of *what survives*, not in terms of which
/// syscall runs, because the syscall differs by platform while the guarantee
/// does not. That is also why there is no separate "plain `fsync`" level: on
/// Linux `fsync` is the power-loss barrier, while on macOS it is not (only
/// `F_FULLFSYNC` flushes the drive cache), so such a level could not be given
/// one honest description.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DurabilityLevel {
    /// No write-ahead log. Nothing survives beyond the caller's most recent
    /// `save()` checkpoint.
    Off,
    /// Log every commit, but do not barrier. An acknowledged mutation
    /// survives the **process** dying — `SIGKILL`, an unhandled panic, an
    /// OOM-kill — because the frame is already in the kernel's page cache.
    /// An OS crash or power loss loses commits made since the last `save()`.
    Normal,
    /// Log every commit and barrier before returning. An acknowledged
    /// mutation survives **power loss**. The default, and the strongest
    /// guarantee the platform offers.
    #[default]
    Full,
}

impl DurabilityLevel {
    /// Whether this level writes a WAL at all. `false` only for
    /// [`DurabilityLevel::Off`].
    #[inline]
    pub fn logs(self) -> bool {
        !matches!(self, Self::Off)
    }

    /// How the WAL should make each frame durable, or `None` when this level
    /// keeps no log. Total by construction, so a new level cannot be added
    /// without deciding its sync behaviour.
    #[inline]
    pub fn sync_mode(self) -> Option<SyncMode> {
        match self {
            Self::Off => None,
            Self::Normal => Some(SyncMode::PageCache),
            Self::Full => Some(SyncMode::Barrier),
        }
    }

    /// The level named by a binding-facing string (`"full"` / `"normal"` /
    /// `"off"`), or `None` if unrecognised. Shared by every binding so the
    /// vocabulary cannot drift between them; the caller owns the error type
    /// and message.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "full" => Some(Self::Full),
            "normal" => Some(Self::Normal),
            "off" => Some(Self::Off),
            _ => None,
        }
    }

    /// The canonical name of this level, the inverse of [`Self::from_name`].
    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Normal => "normal",
            Self::Full => "full",
        }
    }

    /// Every accepted level name, for error messages that need to list them.
    pub const NAMES: [&'static str; 3] = ["full", "normal", "off"];
}

/// How [`Wal::append`] makes a frame durable. Derived from a
/// [`DurabilityLevel`] via [`DurabilityLevel::sync_mode`]; separate from it so
/// that "no log at all" is unrepresentable on an open WAL file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncMode {
    /// Barrier after every frame — `append` returns only once the bytes are
    /// on stable storage. On Apple targets this is `fcntl(F_FULLFSYNC)`;
    /// elsewhere it is `fdatasync`/`fsync`.
    Barrier,
    /// Hand the frame to the OS and return. Bytes are in the kernel page
    /// cache, which outlives the process but not the kernel.
    PageCache,
}

/// One logical, identity-keyed mutation. See the module docs for why
/// the state-changing shapes are idempotent upserts.
///
/// **Variant order is on-disk format.** Postcard tags variants by
/// declaration index, so a new op must be *appended* — inserting one
/// renumbers its successors and silently misparses every existing WAL.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MutationOp {
    /// Add-or-replace a node identified by `(node_type, id)` with the
    /// full given title + property set.
    UpsertNode {
        node_type: String,
        id: Value,
        title: Value,
        properties: Vec<(String, Value)>,
    },
    /// Remove the node identified by `(node_type, id)`, if present.
    RemoveNode { node_type: String, id: Value },
    /// Add-or-replace the edge `(conn_type, src, tgt)` with the full
    /// given property set. Endpoints are named by their logical
    /// `(node_type, id)`.
    UpsertEdge {
        conn_type: String,
        src_type: String,
        src_id: Value,
        tgt_type: String,
        tgt_id: Value,
        properties: Vec<(String, Value)>,
    },
    /// Remove the edge `(conn_type, src, tgt)`, if present.
    RemoveEdge {
        conn_type: String,
        src_type: String,
        src_id: Value,
        tgt_type: String,
        tgt_id: Value,
    },
    /// Replace the **secondary** labels of `(node_type, id)` with exactly
    /// `labels` (the primary type is `node_type` and is never listed).
    ///
    /// A node's secondary labels live in `DirGraph::secondary_label_index`,
    /// *above* the storage backend — `NodeData` carries none — so they are
    /// invisible to the `GraphWrite` capture seam that produces
    /// [`MutationOp::UpsertNode`]. Without this op a `:Label` added by
    /// `CREATE (n:A:B)` / `SET n:B` was lost on WAL replay while every
    /// property survived. Labels are therefore captured at their own choke
    /// point ([`crate::graph::dir_graph::DirGraph::add_node_label`] and its
    /// remove sibling) and carried as a whole set, which keeps the op
    /// idempotent like every other: replaying it twice, or over a
    /// checkpoint that already holds some of the labels, converges on the
    /// same state.
    ///
    /// Ordered by label name, matching `DirGraph::node_labels`, so a
    /// recovered graph reports labels in the same order as the graph that
    /// logged them.
    SetNodeLabels {
        node_type: String,
        id: Value,
        labels: Vec<String>,
    },
}

/// One committed mutation operation: the ops it produced, tagged with a
/// log-sequence number.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WalFrame {
    /// Log-sequence number, issued by the writer's own monotonic counter —
    /// **not** the graph `version`. Frames replay in ascending `lsn`; on
    /// recovery, frames with `lsn <= DirGraph::checkpoint_lsn` are already
    /// folded into the snapshot and skipped.
    ///
    /// The counter must never restart at a checkpoint: a restarted LSN would
    /// be reused by a post-checkpoint frame, making a stale pre-checkpoint
    /// frame indistinguishable from a fresh one.
    pub lsn: u64,
    /// The logical ops this commit produced, in application order.
    pub ops: Vec<MutationOp>,
}

// ─────────────────────────────────────────────────────────────────────
// CRC32 (IEEE 802.3, polynomial 0xEDB88320) — dependency-free, table-
// backed. Deterministic across processes/builds (unlike DefaultHasher),
// which the torn-frame check relies on.
// ─────────────────────────────────────────────────────────────────────

fn crc32_table() -> &'static [u32; 256] {
    static TABLE: OnceLock<[u32; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [0u32; 256];
        let mut n = 0;
        while n < 256 {
            let mut c = n as u32;
            let mut k = 0;
            while k < 8 {
                c = if c & 1 != 0 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
                k += 1;
            }
            table[n] = c;
            n += 1;
        }
        table
    })
}

/// CRC32 (IEEE) of `data`. Used as the per-frame integrity check.
pub fn crc32(data: &[u8]) -> u32 {
    let table = crc32_table();
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc = table[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

// ─────────────────────────────────────────────────────────────────────
// Write side
// ─────────────────────────────────────────────────────────────────────

/// Write the WAL file header (magic + format version) to a freshly
/// created/truncated WAL. Call once before any [`append_frame`].
pub fn write_header(w: &mut impl Write) -> io::Result<()> {
    write_header_version(w, WAL_FORMAT_VERSION)
}

fn write_header_version(w: &mut impl Write, version: u8) -> io::Result<()> {
    w.write_all(&WAL_MAGIC)?;
    w.write_all(&[version])?;
    Ok(())
}

/// Append one frame: `[len][crc][payload]`. The caller is responsible
/// for `fsync`/`flush` after the append to make it durable — this fn
/// only writes the bytes (so a batch of frames can share one fsync if
/// the caller wants).
///
/// The prefix and payload are assembled into one buffer and emitted with a
/// **single** `write_all`, rather than three. That removes two syscalls from
/// the per-commit path, and — more importantly for
/// [`DurabilityLevel::Normal`] — it shrinks the window in which a process
/// death can leave a torn frame: a `write(2)` cannot be interrupted partway
/// by `SIGKILL`, so a frame that fits in one write is either wholly in the
/// page cache or wholly absent. A short write is still possible in
/// principle, which is why the length/CRC torn-tail check remains the
/// authority rather than an optimisation.
pub fn append_frame(w: &mut impl Write, frame: &WalFrame) -> io::Result<()> {
    append_frame_with_codec(w, frame, crate::serde_codec::CURRENT_CODEC)
}

fn append_frame_with_codec(
    w: &mut impl Write,
    frame: &WalFrame,
    codec: crate::serde_codec::CodecVersion,
) -> io::Result<()> {
    let payload = crate::serde_codec::encode_versioned(codec, frame, MAX_WAL_FRAME_BYTES)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let len = u32::try_from(payload.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "WAL frame exceeds 4 GiB"))?;
    let crc = crc32(&payload);
    let mut framed = Vec::with_capacity(8 + payload.len());
    framed.extend_from_slice(&len.to_le_bytes());
    framed.extend_from_slice(&crc.to_le_bytes());
    framed.extend_from_slice(&payload);
    w.write_all(&framed)?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Read side
// ─────────────────────────────────────────────────────────────────────

/// Read a fixed-size buffer, mapping a clean OR partial EOF to `None`
/// (both end the frame stream). Any other I/O error propagates.
fn read_exact_opt(r: &mut impl Read, buf: &mut [u8]) -> io::Result<Option<()>> {
    match r.read_exact(buf) {
        Ok(()) => Ok(Some(())),
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(None),
        Err(e) => Err(e),
    }
}

/// Read and validate the WAL header. Returns the format version, or an
/// error if the magic is wrong. An empty reader (0 bytes) is an error —
/// a WAL file should always carry at least a header.
pub fn read_header(r: &mut impl Read) -> io::Result<u8> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if magic != WAL_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a kglite WAL file (bad magic)",
        ));
    }
    let mut ver = [0u8; 1];
    r.read_exact(&mut ver)?;
    Ok(ver[0])
}

/// Read every intact frame from `r`, which must be positioned at the
/// start of the file; `stream_len` is the total byte length of the
/// stream (file size). Reads and validates the header, then frames
/// until a clean EOF or the first torn/corrupt frame (short read,
/// over-long declared length, or CRC mismatch) — that frame and
/// anything after it are discarded, modelling a crash mid-append.
/// Returns the recovered frames in file order.
///
/// `stream_len` bounds the per-frame allocation: a corrupt length
/// prefix can otherwise ask for up to 4 GiB *before* the short read is
/// detected. A declared length larger than the bytes remaining in the
/// stream is provably torn/corrupt and stops recovery without
/// allocating.
///
/// When recovery stops before consuming the whole stream (a torn tail
/// after a crash, or garbage mid-file), a one-line warning reporting
/// how many frames were recovered and the byte offset of the bad frame
/// is printed to stderr — the frames before it are still returned, so
/// the contract (recover everything up to the first bad frame) is
/// unchanged; the failure is just no longer silent.
pub fn read_frames(mut r: impl Read, stream_len: u64) -> io::Result<Vec<WalFrame>> {
    let version = read_header(&mut r)?;
    let codec = wal_codec(version)?;

    let header_len = (WAL_MAGIC.len() + 1) as u64;
    let mut consumed: u64 = header_len;
    let mut frames = Vec::new();
    let stopped_at = loop {
        let frame_start = consumed;
        let mut len_buf = [0u8; 4];
        if read_exact_opt(&mut r, &mut len_buf)?.is_none() {
            // Clean EOF or torn length prefix. Only warn for a torn
            // (partial) prefix; a clean EOF is the normal end.
            break (frame_start != stream_len).then_some(frame_start);
        }
        let mut crc_buf = [0u8; 4];
        if read_exact_opt(&mut r, &mut crc_buf)?.is_none() {
            break Some(frame_start); // torn: length present, crc missing
        }
        consumed += 8;
        let len = u32::from_le_bytes(len_buf) as u64;
        let expected_crc = u32::from_le_bytes(crc_buf);

        if len == 0 {
            // A run of zero bytes — the shape an OS crash leaves when a
            // file's length was extended but its data block never reached
            // the platter, which `DurabilityLevel::Normal` makes reachable.
            // `crc32(b"") == 0`, so a zero prefix would otherwise pass the
            // CRC check as a "valid" empty frame and reach the decoder.
            // `append_frame` can never emit one (the smallest real payload
            // is a two-byte Postcard `lsn` + `ops` pair), so treat it as the
            // torn tail it is — by intent, rather than relying on the
            // decoder to reject it.
            break Some(frame_start);
        }
        if len > stream_len.saturating_sub(consumed) {
            // Declared length exceeds the bytes that exist — torn or
            // corrupt prefix. Stop WITHOUT allocating `len` bytes.
            break Some(frame_start);
        }
        let mut payload = vec![0u8; len as usize];
        if read_exact_opt(&mut r, &mut payload)?.is_none() {
            break Some(frame_start); // torn: payload short
        }
        consumed += len;
        if crc32(&payload) != expected_crc {
            break Some(frame_start); // corrupt/torn payload — stop here
        }
        let limits = crate::serde_codec::DecodeLimits::new(MAX_WAL_FRAME_BYTES, len);
        match crate::serde_codec::decode_exact_with::<WalFrame>(codec, &payload, len, limits) {
            Ok(frame) => frames.push(frame),
            Err(_) => break Some(frame_start), // unparseable — treat as torn
        }
    };
    if let Some(offset) = stopped_at {
        eprintln!(
            "[kglite] WAL recovery stopped at a torn/corrupt frame at byte offset {offset} \
             (of {stream_len}); recovered {} intact frame(s) before it. This is expected \
             after a crash mid-commit; the torn tail is discarded and will be truncated at \
             the next checkpoint.",
            frames.len()
        );
    }
    Ok(frames)
}

/// Codec for a WAL header version, or an error naming what this build can
/// read. Every version in `MIN_READABLE..=CURRENT` shares one codec and one
/// op schema — see [`WAL_FORMAT_VERSION`].
fn wal_codec(version: u8) -> io::Result<crate::serde_codec::CodecVersion> {
    match version {
        MIN_READABLE_WAL_FORMAT_VERSION..=WAL_FORMAT_VERSION => {
            Ok(crate::serde_codec::CodecVersion::PostcardV1)
        }
        1 => Err(crate::graph::io::file::pre_014_bincode_error(
            "WAL format v1",
        )),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported WAL format version {version} (this build reads \
                 v{MIN_READABLE_WAL_FORMAT_VERSION}-v{WAL_FORMAT_VERSION}). \
                 A WAL newer than the binary cannot be replayed safely: open \
                 the graph with a matching kglite build to recover it, or \
                 delete the '-wal' sidecar to discard work committed since \
                 the last save() checkpoint."
            ),
        )),
    }
}

// ─────────────────────────────────────────────────────────────────────
// File handle — session-scoped append log
// ─────────────────────────────────────────────────────────────────────

/// The sidecar WAL path for a `.kgl` checkpoint file: `<path>-wal`. Keeps
/// the WAL adjacent to its checkpoint so one is never found without the
/// other being locatable.
pub fn wal_path(checkpoint: &Path) -> PathBuf {
    let mut s = checkpoint.as_os_str().to_owned();
    s.push("-wal");
    PathBuf::from(s)
}

/// Read every intact frame from the WAL at `path` for crash recovery.
/// A missing file yields no frames (a graph that was never mutated since
/// its checkpoint). Stops at the first torn/corrupt frame (see
/// [`read_frames`]).
pub fn recover(path: &Path) -> io::Result<Vec<WalFrame>> {
    match File::open(path) {
        Ok(f) => {
            let len = f.metadata()?.len();
            read_frames(BufReader::new(f), len)
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(e),
    }
}

/// Best-effort fsync of a file's parent directory, so a freshly created
/// file's directory entry survives an OS/power crash (mirrors the
/// directory-fsync step of `io/file.rs::write_kgl_with`). Errors are
/// ignored: some filesystems don't support directory fsync, and the
/// file's own contents are already synced.
fn sync_parent_dir(path: &Path) {
    if let Some(dir) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        if let Ok(dirfile) = File::open(dir) {
            let _ = dirfile.sync_all();
        }
    }
}

/// Truncate a WAL to nothing and lay down a fresh header, `fsync`ing the
/// result. The caller supplies a handle opened for ordinary writing — never
/// the append handle (see [`prepare_wal_file`]).
fn truncate_to_header(file: &mut File) -> io::Result<()> {
    use std::io::{Seek, SeekFrom};
    file.set_len(0)?;
    // The read that classified the header left the cursor mid-file. Without
    // this seek an ordinary (non-append) write would land at that offset and
    // leave a hole in front of the header.
    file.seek(SeekFrom::Start(0))?;
    write_header(file)?;
    file.sync_all()
}

/// Validate the WAL at `path`, creating or repairing its header as needed, so
/// that [`Wal::open`] can take an append handle over a file already known to
/// be well-formed.
///
/// All header maintenance happens here, on an ordinary read/write handle, and
/// finishes before the append handle exists. **An append handle is not a
/// general-purpose write handle.** Rust maps `OpenOptions::append(true)` to
/// `FILE_GENERIC_WRITE & !FILE_WRITE_DATA` on Windows — deliberately dropping
/// the very right that truncation and in-place rewrites require — and an
/// append handle ignores seeks on write on every platform. The previous code
/// repaired torn headers through the append handle, which POSIX tolerates and
/// Windows does not; `upgrade_header_version` had already been forced to open
/// its own handle for the same reason, and its job is now folded in here.
///
/// Classification is unchanged: a file too short for a header, or exactly
/// header-sized with the wrong magic, is crash residue that can hold no frame
/// and is repaired in place; a *longer* file with a bad magic could be
/// somebody's data and is refused; an unreadable version is rejected and a
/// readable older one is upgraded.
fn prepare_wal_file(path: &Path) -> io::Result<()> {
    use std::io::{Seek, SeekFrom};
    let header_len = (WAL_MAGIC.len() + 1) as u64;
    let mut file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(path)?;
    let file_len = file.metadata()?.len();

    if file_len == 0 {
        write_header(&mut file)?;
        file.sync_all()?;
        // fsync the parent directory so the file's creation itself survives a
        // crash (same doctrine as io/file.rs's atomic save: without this the
        // fsync'd file can vanish with the unsynced directory entry).
        sync_parent_dir(path);
        return Ok(());
    }

    let mut header = [0u8; 5];
    let read_len = file_len.min(header_len) as usize;
    file.read_exact(&mut header[..read_len])?;
    let magic_ok = read_len >= WAL_MAGIC.len() && header[..4] == WAL_MAGIC;

    if file_len < header_len || (!magic_ok && file_len == header_len) {
        return truncate_to_header(&mut file);
    }
    if !magic_ok {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} is not a kglite WAL file (bad magic) and is not empty; \
                 refusing to overwrite it. Move the file aside if it is stale.",
                path.display()
            ),
        ));
    }

    // Reject an unreadable version before appending to it; the codec lookup
    // owns the actionable message.
    wal_codec(header[4])?;
    if header[4] != WAL_FORMAT_VERSION {
        // A readable older version. We are about to append current-format
        // frames, so the header must advertise the newer version or a future
        // reader would parse the new frames under the old schema. Rewriting
        // the byte is lossless precisely because the older format is a subset
        // (see `WAL_FORMAT_VERSION`): the frames already in the file are valid
        // current-format frames, and the per-frame CRCs cover payloads only,
        // not the header.
        file.seek(SeekFrom::Start(WAL_MAGIC.len() as u64))?;
        file.write_all(&[WAL_FORMAT_VERSION])?;
        file.sync_data()?;
    }
    Ok(())
}

/// An open, append-only WAL file. Session-scoped (one per open graph
/// file) — it owns a `File` handle, so it lives *outside* the CoW-cloned
/// `DirGraph` (which must stay `Clone`). Each [`append`](Self::append)
/// writes a frame, and under [`SyncMode::Barrier`] also flushes it to
/// stable storage, making the committed mutation durable before the call
/// returns.
///
/// The handle is deliberately **unbuffered** — `file` is a bare [`File`],
/// never a `BufWriter`. That is what makes [`SyncMode::PageCache`] mean
/// anything: the bytes are in the kernel's page cache by the time `append`
/// returns, so they outlive the process even without a barrier. Wrapping
/// this in a userspace buffer would silently downgrade
/// [`DurabilityLevel::Normal`] to "survives nothing".
#[derive(Debug)]
pub struct Wal {
    file: File,
    path: PathBuf,
    sync: SyncMode,
}

impl Wal {
    /// Open the WAL at `path` for appending, creating it with a fresh
    /// header if absent. An existing WAL is opened in append mode with its
    /// frames intact — call [`recover`] *before* opening if you need to
    /// replay them.
    ///
    /// The header is validated on open. A file too short to hold a full
    /// header, or a header-sized file with the wrong magic, can never
    /// contain a frame — it is the residue of a crash between `create`
    /// and the header `fsync` — so it is truncated and re-initialised in
    /// place. A *longer* file with a bad magic could be somebody's data:
    /// that errors loudly instead of destroying it. A header naming a
    /// version this build cannot read (pre-0.14 v1, or anything newer than
    /// [`WAL_FORMAT_VERSION`]) is rejected before a single frame is
    /// appended; a *readable* older version is upgraded in place, since the
    /// frames already present parse under the current schema unchanged.
    ///
    /// `sync` fixes the per-append durability behaviour for the life of the
    /// handle; see [`SyncMode`]. Header maintenance always barriers
    /// regardless of the level — a WAL whose header might not exist after a
    /// crash could not be recovered at all, and it is paid once per open
    /// rather than once per commit.
    pub fn open(path: PathBuf, sync: SyncMode) -> io::Result<Self> {
        // Create/validate/repair the header first, on an ordinary handle. By
        // the time the append handle below exists the file is well-formed, so
        // that handle only ever has to do what append handles can portably do.
        prepare_wal_file(&path)?;
        let file = OpenOptions::new().read(true).append(true).open(&path)?;
        Ok(Self { file, path, sync })
    }

    /// Append one frame — the commit point.
    ///
    /// Under [`SyncMode::Barrier`] this returns only after the bytes are on
    /// stable storage. Under [`SyncMode::PageCache`] it returns once the
    /// kernel has the bytes, which is the whole of
    /// [`DurabilityLevel::Normal`]'s guarantee: the page cache is the
    /// kernel's, so it survives this process dying but not the kernel
    /// dying.
    pub fn append(&mut self, frame: &WalFrame) -> io::Result<()> {
        append_frame(&mut self.file, frame)?;
        self.file.flush()?;
        if self.sync == SyncMode::Barrier {
            self.file.sync_data()?;
        }
        Ok(())
    }

    /// Flush every frame appended so far to stable storage — the barrier
    /// that [`SyncMode::Barrier`] performs on every commit, on demand.
    ///
    /// Two callers, and both matter:
    ///
    /// 1. **Before a checkpoint.** A checkpoint truncates the log, so the
    ///    frames it folds in must already be on disk. If they are not, an OS
    ///    crash in the window between writing the checkpoint and truncating
    ///    the log can leave a *prefix* of the frames, and replaying that
    ///    prefix over the newer checkpoint would revert data the checkpoint
    ///    already holds. Under [`SyncMode::Barrier`] the frames are on disk
    ///    already and this is the no-op it looks like; under
    ///    [`SyncMode::PageCache`] it is load-bearing.
    /// 2. **On user demand.** It is the only way a `Normal` graph can reach
    ///    power-safety at a granularity finer than a whole checkpoint —
    ///    "flush at end of request", "flush before shutdown".
    pub fn sync(&mut self) -> io::Result<()> {
        self.file.flush()?;
        self.file.sync_data()
    }

    /// Reset to an empty WAL (header only), `fsync`ing the truncation.
    /// Called after a checkpoint (a full `.kgl` save) has folded every
    /// frame into the snapshot, so the log can start fresh.
    pub fn reset(&mut self) -> io::Result<()> {
        // Truncation and header rewrite go through a dedicated read/write
        // handle for the same reason `prepare_wal_file` does: the append
        // handle lacks the write-data right on Windows and ignores seeks on
        // write everywhere. `self.file` stays usable afterwards — append mode
        // resolves the end of the file at write time, so the next frame lands
        // straight after the fresh header.
        let mut file = OpenOptions::new().read(true).write(true).open(&self.path)?;
        truncate_to_header(&mut file)
    }

    /// The WAL's filesystem path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use tempfile::TempDir;

    /// Deliberately **v2-only** ops (no `SetNodeLabels`): these double as
    /// the fixture for `v2_frames_replay_exactly_under_current_schema`,
    /// which is only meaningful if every op in it predates v3.
    fn sample_ops() -> Vec<MutationOp> {
        vec![
            MutationOp::UpsertNode {
                node_type: "Person".to_string(),
                id: Value::Int64(1),
                title: Value::String("Alice".to_string()),
                properties: vec![
                    ("age".to_string(), Value::Int64(30)),
                    ("city".to_string(), Value::String("Oslo".to_string())),
                ],
            },
            MutationOp::UpsertEdge {
                conn_type: "KNOWS".to_string(),
                src_type: "Person".to_string(),
                src_id: Value::Int64(1),
                tgt_type: "Person".to_string(),
                tgt_id: Value::Int64(2),
                properties: vec![("since".to_string(), Value::Int64(2020))],
            },
            MutationOp::RemoveNode {
                node_type: "Person".to_string(),
                id: Value::Int64(9),
            },
        ]
    }

    fn write_wal(frames: &[WalFrame]) -> Vec<u8> {
        write_wal_version(frames, WAL_FORMAT_VERSION)
    }

    fn write_wal_version(frames: &[WalFrame], version: u8) -> Vec<u8> {
        let mut buf = Vec::new();
        write_header_version(&mut buf, version).unwrap();
        let codec = wal_codec(version).unwrap();
        for f in frames {
            append_frame_with_codec(&mut buf, f, codec).unwrap();
        }
        buf
    }

    /// Test shim: run [`read_frames`] over an in-memory byte buffer,
    /// supplying its length as the stream length (as `recover` does
    /// with the file size).
    fn read_frames_all(bytes: Vec<u8>) -> io::Result<Vec<WalFrame>> {
        let len = bytes.len() as u64;
        read_frames(Cursor::new(bytes), len)
    }

    /// Open a WAL at the full barrier — the default level, and the one every
    /// pre-existing test in this module was written against. Tests that care
    /// about the no-barrier rung call [`Wal::open`] directly with
    /// [`SyncMode::PageCache`].
    fn open_wal(path: PathBuf) -> io::Result<Wal> {
        Wal::open(path, SyncMode::Barrier)
    }

    #[test]
    fn crc32_matches_known_vector() {
        // CRC32/IEEE of "123456789" is the standard check value.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn single_frame_round_trips() {
        let frame = WalFrame {
            lsn: 1,
            ops: sample_ops(),
        };
        let bytes = write_wal(std::slice::from_ref(&frame));
        let got = read_frames_all(bytes).unwrap();
        assert_eq!(got, vec![frame]);
    }

    #[test]
    fn multiple_frames_preserve_order() {
        let frames = vec![
            WalFrame {
                lsn: 1,
                ops: vec![MutationOp::RemoveNode {
                    node_type: "T".into(),
                    id: Value::Int64(1),
                }],
            },
            WalFrame {
                lsn: 2,
                ops: sample_ops(),
            },
            WalFrame {
                lsn: 3,
                ops: vec![],
            },
        ];
        let bytes = write_wal(&frames);
        let got = read_frames_all(bytes).unwrap();
        assert_eq!(got, frames);
    }

    #[test]
    fn torn_trailing_frame_is_discarded() {
        let frames = vec![
            WalFrame {
                lsn: 1,
                ops: sample_ops(),
            },
            WalFrame {
                lsn: 2,
                ops: sample_ops(),
            },
        ];
        let mut bytes = write_wal(&frames);
        // Simulate a crash mid-append: lop off the last 5 bytes of the
        // final frame's payload.
        bytes.truncate(bytes.len() - 5);
        let got = read_frames_all(bytes).unwrap();
        // Only the first, fully-written frame survives.
        assert_eq!(got, vec![frames[0].clone()]);
    }

    #[test]
    fn truncated_in_length_prefix_is_clean_stop() {
        let frames = vec![WalFrame {
            lsn: 1,
            ops: sample_ops(),
        }];
        let mut bytes = write_wal(&frames);
        // Append a stray partial length prefix (2 of 4 bytes) — a crash
        // before even the length was fully written.
        bytes.extend_from_slice(&[0u8, 0u8]);
        let got = read_frames_all(bytes).unwrap();
        assert_eq!(got, frames);
    }

    #[test]
    fn corrupt_payload_crc_mismatch_stops() {
        let frame = WalFrame {
            lsn: 1,
            ops: sample_ops(),
        };
        let mut bytes = write_wal(std::slice::from_ref(&frame));
        // Flip a byte in the payload (after the 5-byte header + 8-byte
        // len/crc prefix) — CRC must catch it and drop the frame.
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        let got = read_frames_all(bytes).unwrap();
        assert!(got.is_empty(), "corrupt frame must not be returned");
    }

    #[test]
    fn header_only_wal_yields_no_frames() {
        let bytes = write_wal(&[]);
        let got = read_frames_all(bytes).unwrap();
        assert!(got.is_empty());
    }

    #[test]
    fn bad_magic_is_rejected() {
        let bytes = b"XXXX\x02".to_vec();
        assert!(read_frames_all(bytes).is_err());
    }

    #[test]
    fn legacy_v1_is_rejected_before_frame_recovery() {
        let bytes = b"KWAL\x01".to_vec();
        let error = read_frames_all(bytes).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("pre-0.14"));
    }

    #[test]
    fn unknown_version_is_rejected_without_payload_sniffing() {
        let bytes = b"KWAL\x7f".to_vec();
        let error = read_frames_all(bytes).unwrap_err();
        assert!(error
            .to_string()
            .contains("unsupported WAL format version 127"));
    }

    #[test]
    fn empty_reader_is_error() {
        let bytes: Vec<u8> = Vec::new();
        assert!(read_frames_all(bytes).is_err());
    }

    // ── op-schema stability (v2 ⊂ v3) ────────────────────────────────

    /// Postcard tags enum variants by declaration index, so the tag of
    /// every pre-existing op is on-disk format: renumbering one silently
    /// misparses every WAL ever written. A single-op frame encodes as
    /// `[lsn varint][ops len varint][variant tag varint]…`, so byte 2 is
    /// the tag. Pinning all five keeps a future op from being *inserted*
    /// rather than appended.
    #[test]
    fn variant_tags_are_stable_on_disk_format() {
        let id = || Value::Int64(1);
        let cases: [(u8, MutationOp); 5] = [
            (
                0,
                MutationOp::UpsertNode {
                    node_type: "T".into(),
                    id: id(),
                    title: Value::Null,
                    properties: vec![],
                },
            ),
            (
                1,
                MutationOp::RemoveNode {
                    node_type: "T".into(),
                    id: id(),
                },
            ),
            (
                2,
                MutationOp::UpsertEdge {
                    conn_type: "C".into(),
                    src_type: "T".into(),
                    src_id: id(),
                    tgt_type: "T".into(),
                    tgt_id: id(),
                    properties: vec![],
                },
            ),
            (
                3,
                MutationOp::RemoveEdge {
                    conn_type: "C".into(),
                    src_type: "T".into(),
                    src_id: id(),
                    tgt_type: "T".into(),
                    tgt_id: id(),
                },
            ),
            (
                4,
                MutationOp::SetNodeLabels {
                    node_type: "T".into(),
                    id: id(),
                    labels: vec![],
                },
            ),
        ];
        for (tag, op) in cases {
            let mut buf = Vec::new();
            append_frame(
                &mut buf,
                &WalFrame {
                    lsn: 1,
                    ops: vec![op.clone()],
                },
            )
            .unwrap();
            // Skip the 8-byte [len][crc] prefix, then [lsn=1][ops_len=1].
            assert_eq!(
                buf[8 + 2],
                tag,
                "variant tag for {op:?} moved — this breaks every WAL on disk"
            );
        }
    }

    /// A v2 WAL (written before `SetNodeLabels` existed) must replay
    /// *exactly* under the current schema — no compat mirror, no discarded
    /// frames. This is the upgrade path for a graph that crashed under an
    /// older build.
    #[test]
    fn v2_frames_replay_exactly_under_current_schema() {
        let frames = vec![
            WalFrame {
                lsn: 1,
                ops: sample_ops(),
            },
            WalFrame {
                lsn: 2,
                ops: sample_ops(),
            },
        ];
        let bytes = write_wal_version(&frames, MIN_READABLE_WAL_FORMAT_VERSION);
        assert_eq!(bytes[4], 2, "fixture must carry a v2 header");
        assert_eq!(read_frames_all(bytes).unwrap(), frames);
    }

    /// Opening a readable older WAL for append upgrades its header, so the
    /// current-format frames we are about to write are not later parsed
    /// under the old version. The pre-existing frames survive.
    #[test]
    fn open_upgrades_readable_older_header_and_keeps_frames() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("g.kgl-wal");
        std::fs::write(
            &p,
            write_wal_version(&[frame(1)], MIN_READABLE_WAL_FORMAT_VERSION),
        )
        .unwrap();

        let mut wal = open_wal(p.clone()).unwrap();
        wal.append(&WalFrame {
            lsn: 2,
            ops: vec![MutationOp::SetNodeLabels {
                node_type: "Person".into(),
                id: Value::Int64(1),
                labels: vec!["Employee".into()],
            }],
        })
        .unwrap();
        drop(wal);

        assert_eq!(
            std::fs::read(&p).unwrap()[4],
            WAL_FORMAT_VERSION,
            "header must be upgraded before newer frames are appended"
        );
        let got = recover(&p).unwrap();
        assert_eq!(got.iter().map(|f| f.lsn).collect::<Vec<_>>(), [1, 2]);
        assert_eq!(got[0], frame(1), "the pre-upgrade frame is unchanged");
    }

    /// A WAL from a *newer* build must be refused loudly rather than
    /// silently truncated to the frames this build happens to parse.
    #[test]
    fn newer_wal_is_refused_with_actionable_message() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("g.kgl-wal");
        // Header only, hand-built: this build cannot encode frames for a
        // version it does not know.
        let mut header = WAL_MAGIC.to_vec();
        header.push(WAL_FORMAT_VERSION + 1);
        std::fs::write(&p, &header).unwrap();
        for message in [
            open_wal(p.clone()).unwrap_err().to_string(),
            recover(&p).unwrap_err().to_string(),
        ] {
            assert!(
                message.contains("unsupported WAL format version"),
                "{message}"
            );
            assert!(message.contains("matching kglite build"), "{message}");
        }
    }

    // ── file handle ──────────────────────────────────────────────────

    fn frame(lsn: u64) -> WalFrame {
        WalFrame {
            lsn,
            ops: sample_ops(),
        }
    }

    #[test]
    fn open_creates_with_header_and_appends_survive_reopen() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("g.kgl-wal");
        {
            let mut wal = open_wal(p.clone()).unwrap();
            wal.append(&frame(1)).unwrap();
            wal.append(&frame(2)).unwrap();
        } // drop closes the file
          // Reopen for append (must NOT clobber existing frames)...
        {
            let mut wal = open_wal(p.clone()).unwrap();
            wal.append(&frame(3)).unwrap();
        }
        let frames = recover(&p).unwrap();
        assert_eq!(frames.iter().map(|f| f.lsn).collect::<Vec<_>>(), [1, 2, 3]);
    }

    #[test]
    fn open_rejects_legacy_wal_before_append() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("g.kgl-wal");
        std::fs::write(&p, b"KWAL\x01").unwrap();

        let error = open_wal(p).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn reset_truncates_to_header_only() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("g.kgl-wal");
        let mut wal = open_wal(p.clone()).unwrap();
        wal.append(&frame(1)).unwrap();
        wal.append(&frame(2)).unwrap();
        wal.reset().unwrap();
        assert!(recover(&p).unwrap().is_empty());
        // Still usable after reset.
        wal.append(&frame(5)).unwrap();
        assert_eq!(
            recover(&p)
                .unwrap()
                .iter()
                .map(|f| f.lsn)
                .collect::<Vec<_>>(),
            [5]
        );
    }

    #[test]
    fn recover_missing_file_is_empty() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("does-not-exist.kgl-wal");
        assert!(recover(&p).unwrap().is_empty());
    }

    #[test]
    fn wal_path_appends_suffix() {
        assert_eq!(
            wal_path(Path::new("/data/graph.kgl")),
            PathBuf::from("/data/graph.kgl-wal")
        );
    }

    // ── hardening: torn header / corrupt length / bad magic ─────────

    /// A crash between `File::create` and the header fsync leaves a
    /// 0–4 byte file. `open` must repair it (truncate + rewrite the
    /// header) and the WAL must be fully usable afterwards.
    #[test]
    fn open_repairs_torn_header() {
        for torn_len in 0..5usize {
            let dir = TempDir::new().unwrap();
            let p = dir.path().join("g.kgl-wal");
            std::fs::write(&p, &WAL_MAGIC[..torn_len.min(4)]).unwrap();
            // For torn_len == 4 the magic is complete but the version
            // byte is missing — still shorter than a full header.
            let mut wal = open_wal(p.clone()).unwrap();
            wal.append(&frame(1)).unwrap();
            drop(wal);
            let frames = recover(&p).unwrap();
            assert_eq!(
                frames.iter().map(|f| f.lsn).collect::<Vec<_>>(),
                [1],
                "torn header of {torn_len} bytes must be repaired"
            );
        }
    }

    /// A header-sized file with the wrong magic can hold no frames —
    /// repair it too (crash could sync garbage for the header page).
    #[test]
    fn open_repairs_header_sized_bad_magic() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("g.kgl-wal");
        std::fs::write(&p, b"XXXXX").unwrap();
        let mut wal = open_wal(p.clone()).unwrap();
        wal.append(&frame(7)).unwrap();
        drop(wal);
        assert_eq!(recover(&p).unwrap().len(), 1);
    }

    /// A bad-magic file with MORE than a header's worth of data could
    /// be someone's data — `open` must refuse, not destroy it.
    #[test]
    fn open_refuses_bad_magic_with_data() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("g.kgl-wal");
        std::fs::write(&p, b"not a wal file at all").unwrap();
        let err = open_wal(p.clone()).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        // The file is untouched.
        assert_eq!(std::fs::read(&p).unwrap(), b"not a wal file at all");
    }

    /// A corrupt length prefix must not drive a multi-GiB allocation:
    /// the declared length is capped against the stream size, so a
    /// 0xFFFF_FFFF prefix on a tiny file ends recovery gracefully with
    /// the intact frames — asserted via recovered count, not by
    /// probing the allocator.
    #[test]
    fn corrupt_giant_length_prefix_is_bounded() {
        let frames = vec![frame(1), frame(2)];
        let mut bytes = write_wal(&frames);
        // Append a "frame" whose length prefix claims ~4 GiB.
        bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // len
        bytes.extend_from_slice(&0xDEAD_BEEFu32.to_le_bytes()); // crc
        bytes.extend_from_slice(b"tiny tail, nowhere near 4 GiB");
        let got = read_frames_all(bytes).unwrap();
        assert_eq!(got, frames, "intact frames before the bad prefix survive");
    }

    /// Garbage mid-file: recovery stops at the first bad frame and
    /// returns everything before it (existing contract, locked in).
    #[test]
    fn garbage_mid_file_stops_at_first_bad_frame() {
        let good = vec![frame(1), frame(2)];
        let mut bytes = write_wal(&good);
        // A structurally-plausible but corrupt frame (bad CRC), then a
        // perfectly valid frame after it.
        let mut corrupt = Vec::new();
        append_frame(&mut corrupt, &frame(3)).unwrap();
        corrupt[10] ^= 0xFF; // flip a payload byte, CRC now mismatches
        bytes.extend_from_slice(&corrupt);
        append_frame(&mut bytes, &frame(4)).unwrap();
        let got = read_frames_all(bytes).unwrap();
        // Frames 1-2 recovered; 3 is corrupt; 4 is unreachable (a
        // frame boundary can't be trusted past corruption).
        assert_eq!(got.iter().map(|f| f.lsn).collect::<Vec<_>>(), [1, 2]);
    }

    /// `Wal::open` on a fresh path must leave a recoverable, valid WAL
    /// even before any append (header fsync + parent dir fsync).
    #[test]
    fn open_fresh_file_is_immediately_recoverable() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("g.kgl-wal");
        let _wal = open_wal(p.clone()).unwrap();
        assert!(recover(&p).unwrap().is_empty());
    }

    // ── durability levels ────────────────────────────────────────────

    /// The level → sync-mode mapping is the whole of the feature, so pin it
    /// rather than trusting the match arms to stay put.
    #[test]
    fn level_maps_to_sync_mode_and_round_trips_by_name() {
        assert_eq!(DurabilityLevel::Off.sync_mode(), None);
        assert_eq!(
            DurabilityLevel::Normal.sync_mode(),
            Some(SyncMode::PageCache)
        );
        assert_eq!(DurabilityLevel::Full.sync_mode(), Some(SyncMode::Barrier));

        assert!(!DurabilityLevel::Off.logs());
        assert!(DurabilityLevel::Normal.logs());
        assert!(DurabilityLevel::Full.logs());

        // The default must stay `Full`: weakening it is a maintainer
        // decision, never a side effect of editing this enum.
        assert_eq!(DurabilityLevel::default(), DurabilityLevel::Full);

        for name in DurabilityLevel::NAMES {
            let level = DurabilityLevel::from_name(name).expect("listed name must parse");
            assert_eq!(level.name(), name);
        }
        assert_eq!(DurabilityLevel::from_name("fsync"), None);
        assert_eq!(DurabilityLevel::from_name("FULL"), None);
    }

    /// The `Normal` rung's core claim at the format level: a frame appended
    /// without a barrier is still a complete, recoverable frame. (This test
    /// cannot observe the *absence* of the fsync — that is what the
    /// process-crash tests in `tests/test_durability.py` are for. What it
    /// pins is that skipping the barrier does not corrupt or truncate.)
    #[test]
    fn page_cache_appends_are_recoverable() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("g.kgl-wal");
        {
            let mut wal = Wal::open(p.clone(), SyncMode::PageCache).unwrap();
            wal.append(&frame(1)).unwrap();
            wal.append(&frame(2)).unwrap();
        }
        let got = recover(&p).unwrap();
        assert_eq!(got.iter().map(|f| f.lsn).collect::<Vec<_>>(), [1, 2]);
    }

    /// `sync()` is callable at every mode and leaves the log intact — under
    /// `Barrier` it is redundant, under `PageCache` it is the user-facing
    /// route to power-safety without a full checkpoint.
    #[test]
    fn explicit_sync_preserves_frames_at_every_mode() {
        for mode in [SyncMode::Barrier, SyncMode::PageCache] {
            let dir = TempDir::new().unwrap();
            let p = dir.path().join("g.kgl-wal");
            let mut wal = Wal::open(p.clone(), mode).unwrap();
            wal.append(&frame(1)).unwrap();
            wal.sync().unwrap();
            wal.append(&frame(2)).unwrap();
            wal.sync().unwrap();
            drop(wal);
            assert_eq!(
                recover(&p)
                    .unwrap()
                    .iter()
                    .map(|f| f.lsn)
                    .collect::<Vec<_>>(),
                [1, 2],
                "sync() must not disturb the log at {mode:?}"
            );
        }
    }

    /// A zero-filled run is what an OS crash leaves when a file's length was
    /// extended but its data block never landed — reachable only once the
    /// per-commit barrier is optional. `crc32(b"") == 0`, so without the
    /// explicit guard a zero prefix passes the CRC check as a "valid" empty
    /// frame, and only the decoder's failure stops recovery. Stop by intent.
    #[test]
    fn zero_filled_hole_is_treated_as_a_torn_tail() {
        let good = vec![frame(1), frame(2)];
        let mut bytes = write_wal(&good);
        // A zero-length/zero-CRC prefix: self-consistent, and not a frame.
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        // A perfectly valid frame after the hole must stay unreachable — a
        // frame boundary cannot be trusted past a gap.
        append_frame(&mut bytes, &frame(3)).unwrap();

        let got = read_frames_all(bytes).unwrap();
        assert_eq!(got.iter().map(|f| f.lsn).collect::<Vec<_>>(), [1, 2]);
    }

    /// A whole page of zeros — the realistic shape of the hazard above.
    #[test]
    fn zero_page_after_frames_recovers_the_prefix() {
        let mut bytes = write_wal(&[frame(1)]);
        bytes.extend_from_slice(&[0u8; 4096]);
        let got = read_frames_all(bytes).unwrap();
        assert_eq!(got, vec![frame(1)]);
    }

    /// Counts `write` calls so the single-syscall property is asserted, not
    /// assumed. `write_all` issues exactly one `write` per full acceptance.
    struct CountingWriter {
        inner: Vec<u8>,
        writes: usize,
    }

    impl Write for CountingWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.writes += 1;
            self.inner.extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    /// One frame is one write. Beyond saving two syscalls per commit, this
    /// is what keeps a `SIGKILL` from landing *between* a frame's length
    /// prefix and its payload: a `write(2)` is not interruptible partway.
    #[test]
    fn frame_is_emitted_in_a_single_write() {
        let mut w = CountingWriter {
            inner: Vec::new(),
            writes: 0,
        };
        append_frame(&mut w, &frame(1)).unwrap();
        assert_eq!(w.writes, 1, "a frame must not be split across writes");

        // …and the bytes are still exactly what the reader expects.
        let mut bytes = Vec::new();
        write_header(&mut bytes).unwrap();
        bytes.extend_from_slice(&w.inner);
        assert_eq!(read_frames_all(bytes).unwrap(), vec![frame(1)]);
    }
}
