//! Write-ahead log for durable in-memory graphs (Stage 1 of the
//! embedded-Cypher-DB durability work).
//!
//! ## What this is
//!
//! A `.kgl-wal` sidecar holds an append-only sequence of **logical**
//! mutation frames. Each committed mutation operation appends one
//! [`WalFrame`] — a batch of [`MutationOp`]s tagged with the post-commit
//! graph `version` as its log-sequence number (LSN) — and `fsync`s. On
//! open, the engine loads the `.kgl` checkpoint snapshot, then replays
//! every WAL frame with `lsn > checkpoint.version` to recover work
//! committed since the last checkpoint. A checkpoint (a full `.kgl`
//! save) truncates the WAL.
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
//! A frame is `[len: u32 LE][crc32: u32 LE][payload: bincode(WalFrame)]`.
//! A crash mid-append leaves a torn trailing frame; [`read_frames`] stops
//! at the first short read or CRC mismatch and returns every frame up to
//! it. A torn frame is therefore *discarded*, never half-applied — the
//! atomic unit of durability is the whole frame, committed by the `fsync`
//! that follows its append.

use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::datatypes::Value;

/// File magic for a kglite WAL sidecar: `KWAL`.
pub const WAL_MAGIC: [u8; 4] = *b"KWAL";

/// On-disk WAL format version. Bumped only on a breaking frame-layout
/// change; the WAL is a within-version recovery artefact (truncated at
/// every checkpoint), not a long-term archival format like `.kgl`.
pub const WAL_FORMAT_VERSION: u8 = 1;

/// One logical, identity-keyed mutation. See the module docs for why
/// the state-changing shapes are idempotent upserts.
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
}

/// One committed mutation operation: the ops it produced, tagged with
/// the post-commit graph version as the log-sequence number.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WalFrame {
    /// Post-commit graph `version`. Frames replay in ascending `lsn`;
    /// on recovery, frames with `lsn <= checkpoint_version` are already
    /// folded into the snapshot and skipped.
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
    w.write_all(&WAL_MAGIC)?;
    w.write_all(&[WAL_FORMAT_VERSION])?;
    Ok(())
}

/// Append one frame: `[len][crc][payload]`. The caller is responsible
/// for `fsync`/`flush` after the append to make it durable — this fn
/// only writes the bytes (so a batch of frames can share one fsync if
/// the caller wants).
pub fn append_frame(w: &mut impl Write, frame: &WalFrame) -> io::Result<()> {
    let payload = crate::serde_codec::encode(frame)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let len = u32::try_from(payload.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "WAL frame exceeds 4 GiB"))?;
    let crc = crc32(&payload);
    w.write_all(&len.to_le_bytes())?;
    w.write_all(&crc.to_le_bytes())?;
    w.write_all(&payload)?;
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
    if version != WAL_FORMAT_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported WAL format version {version} (this build writes \
                 v{WAL_FORMAT_VERSION}); checkpoint with an older build first"
            ),
        ));
    }

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
        match crate::serde_codec::decode::<WalFrame>(&payload) {
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

/// An open, append-only WAL file. Session-scoped (one per open graph
/// file) — it owns a `File` handle, so it lives *outside* the CoW-cloned
/// `DirGraph` (which must stay `Clone`). Each [`append`](Self::append)
/// writes a frame and `fsync`s, making the committed mutation durable
/// before the call returns.
#[derive(Debug)]
pub struct Wal {
    file: File,
    path: PathBuf,
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
    /// that errors loudly instead of destroying it. (`recover` handles
    /// version mismatches; open only repairs what is provably frameless.)
    pub fn open(path: PathBuf) -> io::Result<Self> {
        let existed = path.exists();
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&path)?;
        let header_len = (WAL_MAGIC.len() + 1) as u64;
        let file_len = file.metadata()?.len();
        if !existed || file_len == 0 {
            write_header(&mut file)?;
            file.sync_all()?;
            // fsync the parent directory so the file's creation itself
            // survives a crash (same doctrine as io/file.rs's atomic
            // save: without this the fsync'd file can vanish with the
            // unsynced directory entry).
            sync_parent_dir(&path);
        } else {
            let mut header = [0u8; 5];
            let read_len = file_len.min(header_len) as usize;
            {
                use std::io::{Read, Seek, SeekFrom};
                // `append` mode only affects writes; reads may seek.
                file.seek(SeekFrom::Start(0))?;
                file.read_exact(&mut header[..read_len])?;
                file.seek(SeekFrom::End(0))?;
            }
            let magic_ok = read_len >= WAL_MAGIC.len() && header[..4] == WAL_MAGIC;
            if file_len < header_len || (!magic_ok && file_len == header_len) {
                // Torn header (crash between create and header fsync):
                // shorter than a header, or exactly header-sized with a
                // bad magic. No frame can exist — repair in place.
                file.set_len(0)?;
                write_header(&mut file)?;
                file.sync_all()?;
            } else if !magic_ok {
                // Bad magic with data after the header position: this is
                // not a torn header — refuse to touch it.
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "{} is not a kglite WAL file (bad magic) and is not empty; \
                         refusing to overwrite it. Move the file aside if it is stale.",
                        path.display()
                    ),
                ));
            }
        }
        Ok(Self { file, path })
    }

    /// Append one frame and `fsync` — the durability point. Returns only
    /// after the bytes are on stable storage.
    pub fn append(&mut self, frame: &WalFrame) -> io::Result<()> {
        append_frame(&mut self.file, frame)?;
        self.file.flush()?;
        self.file.sync_data()?;
        Ok(())
    }

    /// Reset to an empty WAL (header only), `fsync`ing the truncation.
    /// Called after a checkpoint (a full `.kgl` save) has folded every
    /// frame into the snapshot, so the log can start fresh.
    pub fn reset(&mut self) -> io::Result<()> {
        self.file.set_len(0)?;
        write_header(&mut self.file)?;
        self.file.sync_all()?;
        Ok(())
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
        let mut buf = Vec::new();
        write_header(&mut buf).unwrap();
        for f in frames {
            append_frame(&mut buf, f).unwrap();
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
        let bytes = b"XXXX\x01".to_vec();
        assert!(read_frames_all(bytes).is_err());
    }

    #[test]
    fn empty_reader_is_error() {
        let bytes: Vec<u8> = Vec::new();
        assert!(read_frames_all(bytes).is_err());
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
            let mut wal = Wal::open(p.clone()).unwrap();
            wal.append(&frame(1)).unwrap();
            wal.append(&frame(2)).unwrap();
        } // drop closes the file
          // Reopen for append (must NOT clobber existing frames)...
        {
            let mut wal = Wal::open(p.clone()).unwrap();
            wal.append(&frame(3)).unwrap();
        }
        let frames = recover(&p).unwrap();
        assert_eq!(frames.iter().map(|f| f.lsn).collect::<Vec<_>>(), [1, 2, 3]);
    }

    #[test]
    fn reset_truncates_to_header_only() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("g.kgl-wal");
        let mut wal = Wal::open(p.clone()).unwrap();
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
            let mut wal = Wal::open(p.clone()).unwrap();
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
        let mut wal = Wal::open(p.clone()).unwrap();
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
        let err = Wal::open(p.clone()).unwrap_err();
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
        let _wal = Wal::open(p.clone()).unwrap();
        assert!(recover(&p).unwrap().is_empty());
    }
}
