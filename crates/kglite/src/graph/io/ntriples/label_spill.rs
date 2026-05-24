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
        })
    }

    /// Append a `(qnum, label)` pair. Labels longer than `u16::MAX`
    /// (65 535 bytes) are truncated — Wikidata labels are ~10-200 chars
    /// in practice, so this limit is a formality.
    pub fn append(&mut self, qnum: u32, label: &str) -> std::io::Result<()> {
        let bytes = label.as_bytes();
        let len = bytes.len().min(u16::MAX as usize) as u16;
        self.writer.write_all(&qnum.to_le_bytes())?;
        self.writer.write_all(&len.to_le_bytes())?;
        self.writer.write_all(&bytes[..len as usize])?;
        Ok(())
    }

    /// Flush and close the journal. Returns the final on-disk size
    /// in bytes — useful for verbose logging.
    pub fn finish(mut self) -> std::io::Result<u64> {
        self.writer.flush()?;
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
    let mut reader = BufReader::with_capacity(64 * 1024, file);
    let mut result: HashMap<u32, String> = HashMap::with_capacity(wanted.len());

    let mut qbuf = [0u8; 4];
    let mut lenbuf = [0u8; 2];

    loop {
        // Graceful EOF on the qnum read.
        match reader.read_exact(&mut qbuf) {
            Ok(()) => {}
            Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
        reader.read_exact(&mut lenbuf)?;
        let qnum = u32::from_le_bytes(qbuf);
        let len = u16::from_le_bytes(lenbuf) as usize;

        if wanted.contains(&qnum) && len > 0 {
            let mut bytes = vec![0u8; len];
            reader.read_exact(&mut bytes)?;
            // Lossy decode: Wikidata labels are valid UTF-8, but a
            // malformed byte sequence shouldn't fail the whole rename
            // pass. Bad labels end up as the U+FFFD-substituted form.
            result.insert(qnum, String::from_utf8_lossy(&bytes).into_owned());
        } else {
            // Skip without allocating. `seek_relative` bypasses the
            // BufReader buffer when possible — fast on NVMe.
            reader.seek(SeekFrom::Current(len as i64))?;
        }
    }
    Ok(result)
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
