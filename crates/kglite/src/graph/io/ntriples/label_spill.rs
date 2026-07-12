//! Append-only on-disk journal for entity-label → type-name resolution.
//!
//! Before 0.8.9 the `load_ntriples` pipeline kept a
//! `HashMap<u32, String>` of every entity's Q-number → label so that
//! `auto_type` could rename types like `Q5` → `human` in the
//! post-Phase-1 rename pass. On Wikidata (124M entities) that map
//! grew to ~10 GB of heap — enough to push a 16 GB machine into swap
//! and collapse the streaming rate from 1.8M triples/s to 450K/s.
//!
//! This module replaces that in-memory cache with a streaming journal
//! on disk:
//!
//! ```text
//! {spill_dir}/labels.bin    [u32 qnum][u16 len][bytes label] …
//! ```
//!
//! `append(qnum, label)` is a buffered sequential write — zero heap
//! growth during Phase 1. The in-Phase-1 `get` path is dropped
//! entirely; types stay as raw Q-codes until the post-Phase-1 rename
//! step, which scans the journal once and extracts labels **only for
//! the ~88K Q-numbers that actually appear as type names**. Memory
//! footprint for the rename: ~3 MB, same as the rename pass
//! previously consumed anyway.
//!
//! Last-write-wins per qnum — if an entity re-emits its label during
//! the same build, the later value is used (matches the old HashMap
//! overwrite semantics).

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// On-disk record size when length == 0: 4 (qnum) + 2 (length) = 6 bytes.
const HEADER_BYTES: usize = 6;

/// Streaming writer for the label journal. Created once at the start
/// of Phase 1, dropped at the end.
pub struct LabelSpillWriter {
    writer: BufWriter<File>,
    /// Labels longer than `u16::MAX` bytes, truncated on append.
    /// Counted so `finish()` can report them (loader convention:
    /// timestamped eprintln, like `loader.rs`'s `eplog!`).
    truncated: u64,
}

impl LabelSpillWriter {
    /// Create a fresh journal at `path`. Truncates any existing file
    /// so a restart of a failed build doesn't read stale data.
    pub fn new(path: &Path) -> std::io::Result<Self> {
        let file = File::create(path)?;
        Ok(Self {
            // 64 KB buffer amortises syscall cost. Larger doesn't help
            // measurably on NVMe and costs memory against the build's
            // already-tight budget.
            writer: BufWriter::with_capacity(64 * 1024, file),
            truncated: 0,
        })
    }

    /// Append a `(qnum, label)` pair. Labels longer than `u16::MAX`
    /// (65 535 bytes) are truncated **at a char boundary** — Wikidata
    /// labels are ~10-200 chars in practice, so this limit is a
    /// formality, but a byte-exact cut could split a multi-byte code
    /// point and turn the tail character into U+FFFD on the reader's
    /// (deliberately lossy) decode. Truncations are counted and
    /// reported by [`finish`](Self::finish).
    pub fn append(&mut self, qnum: u32, label: &str) -> std::io::Result<()> {
        let bytes = label.as_bytes();
        let len = if bytes.len() > u16::MAX as usize {
            // Floor to the nearest char boundary at or below the cap.
            let mut cut = u16::MAX as usize;
            while cut > 0 && !label.is_char_boundary(cut) {
                cut -= 1;
            }
            self.truncated += 1;
            cut
        } else {
            bytes.len()
        };
        self.writer.write_all(&qnum.to_le_bytes())?;
        self.writer.write_all(&(len as u16).to_le_bytes())?;
        self.writer.write_all(&bytes[..len])?;
        Ok(())
    }

    /// Flush and close the journal. Returns the final on-disk size
    /// in bytes — useful for verbose logging.
    pub fn finish(mut self) -> std::io::Result<u64> {
        self.writer.flush()?;
        if self.truncated > 0 {
            eprintln!(
                "[{}] label journal: truncated {} label(s) longer than {} bytes \
                 (kept the leading bytes, cut at a char boundary)",
                chrono::Local::now().format("%H:%M:%S"),
                self.truncated,
                u16::MAX
            );
        }
        let file = self
            .writer
            .into_inner()
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        let size = file.metadata()?.len();
        file.sync_all()?;
        Ok(size)
    }
}

/// Read the journal once, collecting labels **only** for the Q-numbers
/// in `wanted`. Bytes for unwanted entries are skipped with a single
/// `seek_relative` — no allocation, no UTF-8 check.
///
/// Last-write-wins: later entries overwrite earlier ones for the same
/// `qnum`, matching the old `HashMap::insert` semantics.
///
/// Returns a `HashMap` sized to `wanted.len()` (typically tens of
/// thousands on Wikidata vs. the 124M entries the streaming map
/// previously held).
pub fn read_labels_for(
    path: &Path,
    wanted: &HashSet<u32>,
) -> std::io::Result<HashMap<u32, String>> {
    let file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut reader = BufReader::with_capacity(64 * 1024, file);
    let mut result: HashMap<u32, String> = HashMap::with_capacity(wanted.len());

    let mut qbuf = [0u8; 4];
    let mut lenbuf = [0u8; 2];
    // Byte offset of the record currently being read — for the torn-
    // record warning and the payload-fits-in-file check below.
    let mut offset: u64 = 0;
    let torn = |offset: u64, what: &str| {
        eprintln!(
            "[{}] label journal: torn record at byte offset {offset} ({what}); \
             keeping the {} bytes before it and stopping",
            chrono::Local::now().format("%H:%M:%S"),
            offset
        );
    };

    loop {
        // Clean EOF is 0 bytes read at a record boundary. 1-3 bytes is
        // a TORN record (e.g. a crash mid-append): the data before it
        // is fine — stop reading, but say so. (`read_exact` can't
        // distinguish the two, hence `read_full`.)
        let n = read_full(&mut reader, &mut qbuf)?;
        if n == 0 {
            break; // clean EOF
        }
        if n < qbuf.len() {
            torn(offset, "partial qnum header");
            break;
        }
        let n = read_full(&mut reader, &mut lenbuf)?;
        if n < lenbuf.len() {
            torn(offset, "partial length header");
            break;
        }
        let qnum = u32::from_le_bytes(qbuf);
        let len = u16::from_le_bytes(lenbuf) as usize;

        // A payload that extends past the file is torn too — detect it
        // by arithmetic (the skip path's `seek` would silently succeed
        // past EOF and masquerade as a clean EOF on the next record).
        if offset + HEADER_BYTES as u64 + len as u64 > file_len {
            torn(offset, "payload extends past end of file");
            break;
        }

        if wanted.contains(&qnum) && len > 0 {
            let mut bytes = vec![0u8; len];
            reader.read_exact(&mut bytes)?;
            // Lossy decode: Wikidata labels are valid UTF-8, but a
            // malformed byte sequence shouldn't fail the whole rename
            // pass. Bad labels end up as the U+FFFD-substituted form.
            // (This is deliberate; the WRITER truncates at char
            // boundaries so well-formed input never triggers it.)
            result.insert(qnum, String::from_utf8_lossy(&bytes).into_owned());
        } else {
            // Skip without allocating. `seek_relative` bypasses the
            // BufReader buffer when possible — fast on NVMe.
            reader.seek(SeekFrom::Current(len as i64))?;
        }
        offset += HEADER_BYTES as u64 + len as u64;
    }
    Ok(result)
}

/// `read_exact` that reports HOW MANY bytes it read before EOF instead
/// of collapsing "0 bytes" and "some but not all" into one error —
/// needed to tell a clean end-of-journal from a torn trailing record.
/// Non-EOF I/O errors propagate.
fn read_full(r: &mut impl Read, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match r.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(filled)
}

/// Total record overhead per entry, for size estimation in callers
/// that want to predict disk usage.
#[allow(dead_code)]
pub const fn record_overhead_bytes() -> usize {
    HEADER_BYTES
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_path() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("kglite_label_spill_{}_{}.bin", nanos, seq))
    }

    #[test]
    fn write_then_read_wanted_subset() {
        let path = tmp_path();
        let mut w = LabelSpillWriter::new(&path).unwrap();
        w.append(5, "human").unwrap();
        w.append(76, "Barack Obama").unwrap();
        w.append(20, "Norway").unwrap();
        w.append(42, "Douglas Adams").unwrap();
        let size = w.finish().unwrap();
        assert!(size > 0);

        let wanted: HashSet<u32> = [5, 20].into_iter().collect();
        let got = read_labels_for(&path, &wanted).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got.get(&5).unwrap(), "human");
        assert_eq!(got.get(&20).unwrap(), "Norway");
        assert!(!got.contains_key(&76));
        assert!(!got.contains_key(&42));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn last_write_wins_per_qnum() {
        let path = tmp_path();
        let mut w = LabelSpillWriter::new(&path).unwrap();
        w.append(5, "first").unwrap();
        w.append(5, "second").unwrap();
        w.finish().unwrap();

        let wanted: HashSet<u32> = [5].into_iter().collect();
        let got = read_labels_for(&path, &wanted).unwrap();
        assert_eq!(got.get(&5).unwrap(), "second");

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn empty_wanted_set_skips_all() {
        let path = tmp_path();
        let mut w = LabelSpillWriter::new(&path).unwrap();
        for i in 0..1000 {
            w.append(i, "label").unwrap();
        }
        w.finish().unwrap();

        let wanted = HashSet::new();
        let got = read_labels_for(&path, &wanted).unwrap();
        assert!(got.is_empty());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn empty_journal_reads_empty() {
        let path = tmp_path();
        LabelSpillWriter::new(&path).unwrap().finish().unwrap();

        let wanted: HashSet<u32> = [1, 2, 3].into_iter().collect();
        let got = read_labels_for(&path, &wanted).unwrap();
        assert!(got.is_empty());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn oversized_label_truncates_at_char_boundary() {
        // 4-byte code points; u16::MAX (65535) is not a multiple of 4,
        // so a byte-exact cut WOULD split a character. The writer must
        // floor to the nearest boundary and the read-back must be
        // valid UTF-8 (no U+FFFD from the lossy decode).
        let path = tmp_path();
        let big: String = "\u{10348}".repeat(20_000); // 80,000 bytes
        let mut w = LabelSpillWriter::new(&path).unwrap();
        w.append(1, &big).unwrap();
        w.append(2, "after").unwrap();
        w.finish().unwrap();

        let wanted: HashSet<u32> = [1, 2].into_iter().collect();
        let got = read_labels_for(&path, &wanted).unwrap();
        let label = got.get(&1).unwrap();
        assert!(label.len() <= u16::MAX as usize);
        assert_eq!(label.len() % 4, 0, "cut must land on a 4-byte boundary");
        assert!(
            !label.contains('\u{FFFD}'),
            "char-boundary truncation must survive the lossy decode intact"
        );
        assert!(big.starts_with(label.as_str()));
        // The record after the truncated one is still readable.
        assert_eq!(got.get(&2).unwrap(), "after");

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn torn_trailing_header_stops_cleanly() {
        // A crash mid-append can leave 1-3 bytes of the next record's
        // header. The reader must return every complete record before
        // it, without error.
        let path = tmp_path();
        let mut w = LabelSpillWriter::new(&path).unwrap();
        w.append(5, "human").unwrap();
        w.append(20, "Norway").unwrap();
        w.finish().unwrap();
        // Append 3 stray bytes — a torn qnum.
        {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            f.write_all(&[0xAA, 0xBB, 0xCC]).unwrap();
        }

        let wanted: HashSet<u32> = [5, 20].into_iter().collect();
        let got = read_labels_for(&path, &wanted).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got.get(&5).unwrap(), "human");
        assert_eq!(got.get(&20).unwrap(), "Norway");

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn torn_payload_stops_cleanly_even_when_skipped() {
        // A record whose declared payload extends past EOF is torn.
        // The skip path's `seek` would silently jump past EOF, so the
        // reader must catch this by arithmetic — for both wanted and
        // unwanted (skipped) records.
        let path = tmp_path();
        let mut w = LabelSpillWriter::new(&path).unwrap();
        w.append(5, "human").unwrap();
        w.finish().unwrap();
        // Append a header claiming 100 payload bytes, then only 4.
        {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            f.write_all(&9u32.to_le_bytes()).unwrap();
            f.write_all(&100u16.to_le_bytes()).unwrap();
            f.write_all(b"oops").unwrap();
        }
        for wanted_set in [vec![5u32, 9], vec![5u32]] {
            let wanted: HashSet<u32> = wanted_set.into_iter().collect();
            let got = read_labels_for(&path, &wanted).unwrap();
            assert_eq!(got.len(), 1, "only the intact record survives");
            assert_eq!(got.get(&5).unwrap(), "human");
        }

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn zero_length_labels_handled() {
        let path = tmp_path();
        let mut w = LabelSpillWriter::new(&path).unwrap();
        w.append(1, "").unwrap();
        w.append(2, "real").unwrap();
        w.finish().unwrap();

        let wanted: HashSet<u32> = [1, 2].into_iter().collect();
        let got = read_labels_for(&path, &wanted).unwrap();
        // Empty label not inserted (`len > 0` guard).
        assert!(!got.contains_key(&1));
        assert_eq!(got.get(&2).unwrap(), "real");

        let _ = std::fs::remove_file(path);
    }
}
