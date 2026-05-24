//! Parallel decoder for `.bz2` files (single-stream and multistream).
//!
//! Two parallelism axes:
//!
//! - **Multistream** (pbzip2-produced files; concatenated independent
//!   bz2 streams): scan for stream boundaries, dispatch each stream to
//!   a worker, assemble outputs in stream order. Implemented in this
//!   module.
//! - **Single-stream** (Wikidata's `latest-truthy.nt.bz2`, GNU bzip2
//!   default output): a single bz2 stream consists of independently
//!   compressed blocks separated by bit-aligned 48-bit magics
//!   (`0x314159265359`). Block-level parallelism is delegated to
//!   `bzip2_rs::ParallelDecoderReader`, which scans the bitstream for
//!   block magics and decodes each block on a rayon worker. See
//!   `paolobarbolini/bzip2-rs` (MIT/Apache-2.0).
//!
//! This module exposes a single entry point — [`open`] — that returns a
//! `Box<dyn Read + Send>` and picks the right backend automatically.
//!
//! ## Format notes
//!
//! A bz2 stream begins with a 4-byte file header `BZh[1-9]` followed
//! immediately (byte-aligned) by a 48-bit block magic
//! `0x31 0x41 0x59 0x26 0x53 0x59` (regular block) or
//! `0x17 0x72 0x45 0x38 0x50 0x90` (stream-end marker for an empty
//! stream). Probability of a 10-byte false positive in random payload
//! is ~ 1 / 2^48, so a single byte-by-byte scan is sufficient — no
//! decompress probe needed.
//!
//! ## Format notes
//!
//! A bz2 stream begins with a 4-byte file header `BZh[1-9]` followed
//! immediately (byte-aligned) by a 48-bit block magic
//! `0x31 0x41 0x59 0x26 0x53 0x59` (regular block) or
//! `0x17 0x72 0x45 0x38 0x50 0x90` (stream-end marker for an empty
//! stream). Probability of a 10-byte false positive in random payload
//! is ~ 1 / 2^48, so a single byte-by-byte scan is sufficient — no
//! decompress probe needed.

use bzip2::read::BzDecoder;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

/// Soft ceiling on bytes held in flight across workers and the
/// channel. pbzip2 ties its in-flight count to a memory budget rather
/// than a fixed slot count (`pbzip2.cpp:4356`); we follow the same
/// principle so a pathological compression ratio (lbzip2's `expand.c`
/// notes adversarial inputs ~1.8M:1) can't blow out RSS. The default
/// covers Wikidata-class dumps with room to spare while staying well
/// under the loader's other Phase-1 buffers.
const DEFAULT_BUDGET_BYTES: usize = 256 * 1024 * 1024;

/// Multiplier from compressed → estimated decompressed bytes. bz2 on
/// dense text (Wikidata N-Triples) sits around 6-10×; we use 12 as a
/// conservative upper bound. Used only to pick a slot count — not to
/// limit individual decompressions.
const ESTIMATED_RATIO: u64 = 12;

/// Floor on in-flight slots so we never serialise behaviour even on a
/// tiny budget or huge per-stream estimate.
const MIN_IN_FLIGHT: usize = 4;

/// Open a `.bz2` file for streaming reads. Parallelises decompression
/// across an internal thread pool when the file is multistream
/// (pbzip2 / Wikidata format). Falls back to
/// `bzip2::read::MultiBzDecoder` for single-stream input.
pub fn open(path: &Path) -> io::Result<Box<dyn Read + Send>> {
    let offsets = scan_stream_offsets(path)?;
    let file_size = std::fs::metadata(path)?.len();

    if offsets.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{}: no bz2 stream header found", path.display()),
        ));
    }

    if offsets.len() == 1 {
        // Single-stream path — block-level parallelism via bzip2-rs.
        // The decoder scans the bitstream for the 48-bit block magic
        // (`0x314159265359`), splits at block boundaries, and decodes
        // blocks on rayon workers. `max_preread_len` caps the
        // compressed bytes buffered ahead of the workers; the existing
        // `DEFAULT_BUDGET_BYTES` (256 MiB) gives plenty of slack for
        // ~900 KiB blocks at level 9 while staying bounded on
        // pathological compression ratios.
        let file = File::open(path)?;
        let reader = BufReader::with_capacity(8 * 1024 * 1024, file);
        return Ok(Box::new(bzip2_rs::ParallelDecoderReader::new(
            reader,
            bzip2_rs::RayonThreadPool,
            DEFAULT_BUDGET_BYTES,
        )));
    }

    // Build (start, end) pairs.
    let mut streams = Vec::with_capacity(offsets.len());
    for i in 0..offsets.len() {
        let start = offsets[i];
        let end = offsets.get(i + 1).copied().unwrap_or(file_size);
        streams.push((start, end));
    }

    let sizing = Sizing::compute(file_size, streams.len(), DEFAULT_BUDGET_BYTES);
    Ok(Box::new(ParallelBz2Reader::start(
        path.to_path_buf(),
        streams,
        sizing,
    )))
}

/// Worker count + channel capacity derived from the memory budget.
/// `n_workers + channel_cap` is the maximum number of decompressed
/// streams resident at any moment; the byte ceiling is therefore
/// `(n_workers + channel_cap) × estimated_decompressed_per_stream`.
#[derive(Clone, Copy, Debug)]
struct Sizing {
    n_workers: usize,
    channel_cap: usize,
}

impl Sizing {
    fn compute(file_size: u64, n_streams: usize, budget_bytes: usize) -> Self {
        let ncpu = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        // Reserve two cores for the loader's reader thread and the
        // parser/main thread; that drives our parallelism ceiling
        // separately from the memory ceiling below.
        let cpu_cap = ncpu.saturating_sub(2).max(2).min(n_streams);

        // Memory cap: how many average-sized streams fit in the budget.
        let avg_compressed = if n_streams > 0 {
            (file_size / n_streams as u64).max(1)
        } else {
            1
        };
        let estimated_decompressed = avg_compressed.saturating_mul(ESTIMATED_RATIO).max(1);
        let mem_cap = ((budget_bytes as u64) / estimated_decompressed) as usize;
        let mem_cap = mem_cap.max(MIN_IN_FLIGHT);

        // Workers stay under both ceilings; the rest of the budget
        // becomes channel capacity.
        let n_workers = cpu_cap.min(mem_cap.saturating_sub(2).max(2));
        let channel_cap = mem_cap.saturating_sub(n_workers).max(2);
        Sizing {
            n_workers,
            channel_cap,
        }
    }
}

/// Scan the file for `BZh[1-9]` + 6-byte block magic. Returns absolute
/// byte offsets of every detected stream start, in file order.
fn scan_stream_offsets(path: &Path) -> io::Result<Vec<u64>> {
    const CHUNK: usize = 64 * 1024;
    /// Bytes carried between chunks so a header straddling the boundary
    /// is found. 10 = 4-byte file header + 6-byte block magic.
    const PROBE: usize = 10;
    /// After scanning this much of the file, if we've still only seen the
    /// initial stream header, declare the file single-stream and stop.
    /// Multistream bz2 (pbzip2 / Wikipedia dumps) has headers every
    /// ~1 MB, so a 1 GB probe sees dozens of them — a false negative
    /// here would require a multistream file with a >1 GB *first*
    /// stream, which is not produced by any common tool. Wikidata's
    /// single-stream `latest-truthy.nt.bz2` saves ~55 s of pre-scan
    /// time at this threshold (39.7 GB ÷ 700 MB/s).
    const SINGLE_STREAM_PROBE_BYTES: u64 = 1024 * 1024 * 1024;

    let mut file = File::open(path)?;
    let file_size = file.metadata()?.len();

    let mut offsets = Vec::new();
    let mut buf = vec![0u8; CHUNK + PROBE];
    let mut chunk_start: u64 = 0;

    loop {
        if chunk_start >= file_size {
            break;
        }
        file.seek(SeekFrom::Start(chunk_start))?;
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        let is_last = chunk_start + n as u64 >= file_size;
        // On non-final chunks, only commit matches that fit fully inside
        // [0, n - PROBE] so we don't double-count when the next chunk
        // re-scans the overlap.
        let scan_end = if is_last { n } else { n.saturating_sub(PROBE) };

        let mut i = 0;
        while i + PROBE <= n {
            if i >= scan_end {
                break;
            }
            if is_stream_start(&buf[i..i + PROBE]) {
                offsets.push(chunk_start + i as u64);
            }
            i += 1;
        }

        if is_last {
            break;
        }
        chunk_start += scan_end as u64;
        // Early exit: 1 GB scanned, still only the initial header → trust
        // single-stream and skip the remaining ~39 GB. See the constant's
        // doc-comment for the false-negative reasoning.
        if chunk_start >= SINGLE_STREAM_PROBE_BYTES && offsets.len() <= 1 {
            break;
        }
    }

    Ok(offsets)
}

/// Returns true when `buf[..10]` is a bz2 stream header followed by a
/// recognised block magic.
fn is_stream_start(buf: &[u8]) -> bool {
    if buf.len() < 10 {
        return false;
    }
    if buf[0] != 0x42 || buf[1] != 0x5A || buf[2] != 0x68 {
        return false;
    }
    if !(0x31..=0x39).contains(&buf[3]) {
        return false;
    }
    let block = &buf[4..10];
    block == [0x31, 0x41, 0x59, 0x26, 0x53, 0x59] || block == [0x17, 0x72, 0x45, 0x38, 0x50, 0x90]
}

/// Reader that pulls in-order decompressed chunks off a sync channel
/// and drains each one before requesting the next.
struct ParallelBz2Reader {
    rx: Option<Receiver<io::Result<Vec<u8>>>>,
    workers: Option<Vec<JoinHandle<()>>>,
    cur: Vec<u8>,
    cur_pos: usize,
    eof: bool,
}

impl ParallelBz2Reader {
    fn start(path: PathBuf, streams: Vec<(u64, u64)>, sizing: Sizing) -> Self {
        let (tx, rx) = sync_channel::<io::Result<Vec<u8>>>(sizing.channel_cap);

        let streams = Arc::new(streams);
        let next_idx = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let sender = Arc::new(OrderedSender::new(tx));

        let mut workers = Vec::with_capacity(sizing.n_workers);
        for _ in 0..sizing.n_workers {
            let path = path.clone();
            let streams = Arc::clone(&streams);
            let next_idx = Arc::clone(&next_idx);
            let sender = Arc::clone(&sender);
            workers.push(thread::spawn(move || {
                worker_loop(path, streams, next_idx, sender);
            }));
        }

        Self {
            rx: Some(rx),
            workers: Some(workers),
            cur: Vec::new(),
            cur_pos: 0,
            eof: false,
        }
    }
}

impl Read for ParallelBz2Reader {
    fn read(&mut self, dst: &mut [u8]) -> io::Result<usize> {
        loop {
            if self.cur_pos < self.cur.len() {
                let n = (self.cur.len() - self.cur_pos).min(dst.len());
                dst[..n].copy_from_slice(&self.cur[self.cur_pos..self.cur_pos + n]);
                self.cur_pos += n;
                return Ok(n);
            }
            if self.eof {
                return Ok(0);
            }
            let rx = match self.rx.as_ref() {
                Some(r) => r,
                None => return Ok(0),
            };
            match rx.recv() {
                Ok(Ok(buf)) => {
                    self.cur = buf;
                    self.cur_pos = 0;
                }
                Ok(Err(e)) => {
                    self.eof = true;
                    return Err(e);
                }
                Err(_) => {
                    self.eof = true;
                    return Ok(0);
                }
            }
        }
    }
}

impl Drop for ParallelBz2Reader {
    fn drop(&mut self) {
        // Drop the receiver first so any worker blocked in `tx.send()`
        // wakes with a SendError and exits.
        drop(self.rx.take());
        if let Some(workers) = self.workers.take() {
            for h in workers {
                let _ = h.join();
            }
        }
    }
}

fn worker_loop(
    path: PathBuf,
    streams: Arc<Vec<(u64, u64)>>,
    next_idx: Arc<std::sync::atomic::AtomicUsize>,
    sender: Arc<OrderedSender>,
) {
    loop {
        let idx = next_idx.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if idx >= streams.len() {
            return;
        }
        let (start, end) = streams[idx];
        let result = decompress_stream(&path, start, end);
        if !sender.send_in_order(idx, result) {
            return;
        }
    }
}

fn decompress_stream(path: &Path, start: u64, end: u64) -> io::Result<Vec<u8>> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(start))?;
    let limit = end.saturating_sub(start);
    let limited = file.take(limit);
    let mut decoder = BzDecoder::new(BufReader::new(limited));
    let mut out = Vec::new();
    decoder.read_to_end(&mut out)?;
    Ok(out)
}

/// Channel sender that gates each worker's send on a shared
/// `next_to_send` counter, so output is delivered in stream order
/// regardless of which worker finishes first.
struct OrderedSender {
    state: Mutex<OrderedState>,
    cv: Condvar,
    tx: SyncSender<io::Result<Vec<u8>>>,
}

struct OrderedState {
    next_to_send: usize,
    broken: bool,
}

impl OrderedSender {
    fn new(tx: SyncSender<io::Result<Vec<u8>>>) -> Self {
        Self {
            state: Mutex::new(OrderedState {
                next_to_send: 0,
                broken: false,
            }),
            cv: Condvar::new(),
            tx,
        }
    }

    /// Blocks until it is `idx`'s turn to send, then forwards `data`
    /// through the channel and advances `next_to_send`. Returns false
    /// if the receiver has been dropped (signal to the worker to exit).
    fn send_in_order(&self, idx: usize, data: io::Result<Vec<u8>>) -> bool {
        let mut state = self.state.lock().unwrap();
        loop {
            if state.broken {
                return false;
            }
            if state.next_to_send == idx {
                break;
            }
            state = self.cv.wait(state).unwrap();
        }
        let send_result = self.tx.send(data);
        if send_result.is_ok() {
            state.next_to_send += 1;
        } else {
            state.broken = true;
        }
        self.cv.notify_all();
        send_result.is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bzip2::write::BzEncoder;
    use bzip2::Compression;
    use std::io::Write as IoWrite;
    use tempfile::NamedTempFile;

    fn compress(data: &[u8]) -> Vec<u8> {
        let mut enc = BzEncoder::new(Vec::new(), Compression::fast());
        enc.write_all(data).unwrap();
        enc.finish().unwrap()
    }

    fn write_tmp(bytes: &[u8]) -> NamedTempFile {
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(bytes).unwrap();
        tmp.flush().unwrap();
        tmp
    }

    #[test]
    fn scan_finds_correct_offsets() {
        let p1 = compress(b"one");
        let off2 = p1.len() as u64;
        let p2 = compress(b"two");
        let off3 = off2 + p2.len() as u64;
        let p3 = compress(b"three");

        let mut combined = Vec::new();
        combined.extend_from_slice(&p1);
        combined.extend_from_slice(&p2);
        combined.extend_from_slice(&p3);

        let tmp = write_tmp(&combined);
        let offsets = scan_stream_offsets(tmp.path()).unwrap();
        assert_eq!(offsets, vec![0, off2, off3]);
    }

    #[test]
    fn scan_handles_chunk_boundary() {
        // Build a payload large enough to span the 64 KB scanner chunk
        // boundary; place a stream start near the boundary.
        let big = vec![b'a'; 64 * 1024];
        let p1 = compress(&big);
        let p2 = compress(b"after-boundary");
        let mut combined = p1.clone();
        combined.extend_from_slice(&p2);
        let off2 = p1.len() as u64;

        let tmp = write_tmp(&combined);
        let offsets = scan_stream_offsets(tmp.path()).unwrap();
        assert_eq!(offsets, vec![0, off2]);
    }

    #[test]
    fn single_stream_fallback() {
        let data = b"hello bz2 world";
        let compressed = compress(data);
        let tmp = write_tmp(&compressed);

        let mut reader = open(tmp.path()).unwrap();
        let mut out = Vec::new();
        reader.read_to_end(&mut out).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn multistream_roundtrip() {
        let part1 = b"hello ";
        let part2 = b"world\n";
        let mut combined = compress(part1);
        combined.extend_from_slice(&compress(part2));

        let tmp = write_tmp(&combined);

        let mut reader = open(tmp.path()).unwrap();
        let mut out = Vec::new();
        reader.read_to_end(&mut out).unwrap();
        assert_eq!(out, b"hello world\n");
    }

    #[test]
    fn cross_stream_line_preservation() {
        // Stream 1 ends mid-line; stream 2 finishes it.
        let p1 = b"line1\nincomp";
        let p2 = b"lete\nline3\n";
        let mut combined = compress(p1);
        combined.extend_from_slice(&compress(p2));

        let tmp = write_tmp(&combined);
        let mut reader = open(tmp.path()).unwrap();
        let mut out = String::new();
        reader.read_to_string(&mut out).unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines, vec!["line1", "incomplete", "line3"]);
    }

    #[test]
    fn many_streams_parallel() {
        let mut combined = Vec::new();
        let mut expected = Vec::new();
        for i in 0..32 {
            let part = format!("stream-{:02}-content-line\n", i);
            expected.extend_from_slice(part.as_bytes());
            combined.extend_from_slice(&compress(part.as_bytes()));
        }
        let tmp = write_tmp(&combined);

        let mut reader = open(tmp.path()).unwrap();
        let mut out = Vec::new();
        reader.read_to_end(&mut out).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn empty_file_errors() {
        let tmp = write_tmp(b"");
        assert!(open(tmp.path()).is_err());
    }

    #[test]
    fn random_garbage_errors() {
        let tmp = write_tmp(b"not a bz2 file at all, just garbage bytes 12345");
        assert!(open(tmp.path()).is_err());
    }

    #[test]
    fn sizing_respects_budget_for_large_streams() {
        // 8 streams averaging 100 MB compressed ⇒ ~1.2 GB estimated
        // decompressed each. With a 256 MB budget the in-flight slot
        // count should drop to MIN_IN_FLIGHT, not balloon with cores.
        let s = Sizing::compute(800 * 1024 * 1024, 8, 256 * 1024 * 1024);
        assert!(
            s.n_workers + s.channel_cap <= MIN_IN_FLIGHT.max(s.n_workers + s.channel_cap),
            "in-flight cap respected: {:?}",
            s
        );
        assert!(s.n_workers >= 2);
        assert!(s.channel_cap >= 2);
    }

    #[test]
    fn sizing_uses_full_parallelism_for_small_streams() {
        // 1000 streams averaging 100 KB compressed each ⇒ ~1.2 MB
        // decompressed. 256 MB budget admits >100 in-flight, so we
        // should be CPU-bound rather than memory-bound.
        let s = Sizing::compute(100 * 1024 * 1024, 1000, 256 * 1024 * 1024);
        assert!(s.n_workers >= 2);
        let ncpu = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let expected_cap = ncpu.saturating_sub(2).max(2);
        assert!(s.n_workers <= expected_cap);
    }

    #[test]
    fn truncated_stream_errors_at_read_time() {
        let full = compress(b"some payload that we will then truncate");
        let truncated = &full[..full.len() / 2];
        let tmp = write_tmp(truncated);

        // The header still scans as a stream start, so `open` succeeds.
        let mut reader = open(tmp.path()).unwrap();
        let mut out = Vec::new();
        let result = reader.read_to_end(&mut out);
        assert!(result.is_err(), "truncated stream must error on read");
    }
}
