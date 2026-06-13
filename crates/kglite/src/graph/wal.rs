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

use std::io::{self, Read, Write};
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
    let payload =
        bincode::serialize(frame).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
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
/// start of the file. Reads and validates the header, then frames until
/// a clean EOF or the first torn/corrupt frame (short read or CRC
/// mismatch) — that frame and anything after it are discarded, modelling
/// a crash mid-append. Returns the recovered frames in file order.
pub fn read_frames(mut r: impl Read) -> io::Result<Vec<WalFrame>> {
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

    let mut frames = Vec::new();
    loop {
        let mut len_buf = [0u8; 4];
        if read_exact_opt(&mut r, &mut len_buf)?.is_none() {
            break; // clean EOF or torn length prefix
        }
        let mut crc_buf = [0u8; 4];
        if read_exact_opt(&mut r, &mut crc_buf)?.is_none() {
            break; // torn: header present, crc missing
        }
        let len = u32::from_le_bytes(len_buf) as usize;
        let expected_crc = u32::from_le_bytes(crc_buf);

        let mut payload = vec![0u8; len];
        if read_exact_opt(&mut r, &mut payload)?.is_none() {
            break; // torn: payload short
        }
        if crc32(&payload) != expected_crc {
            break; // corrupt/torn payload — stop here
        }
        match bincode::deserialize::<WalFrame>(&payload) {
            Ok(frame) => frames.push(frame),
            Err(_) => break, // unparseable payload — treat as torn tail
        }
    }
    Ok(frames)
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

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
        let got = read_frames(Cursor::new(bytes)).unwrap();
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
        let got = read_frames(Cursor::new(bytes)).unwrap();
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
        let got = read_frames(Cursor::new(bytes)).unwrap();
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
        let got = read_frames(Cursor::new(bytes)).unwrap();
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
        let got = read_frames(Cursor::new(bytes)).unwrap();
        assert!(got.is_empty(), "corrupt frame must not be returned");
    }

    #[test]
    fn header_only_wal_yields_no_frames() {
        let bytes = write_wal(&[]);
        let got = read_frames(Cursor::new(bytes)).unwrap();
        assert!(got.is_empty());
    }

    #[test]
    fn bad_magic_is_rejected() {
        let bytes = b"XXXX\x01".to_vec();
        assert!(read_frames(Cursor::new(bytes)).is_err());
    }

    #[test]
    fn empty_reader_is_error() {
        let bytes: Vec<u8> = Vec::new();
        assert!(read_frames(Cursor::new(bytes)).is_err());
    }
}
