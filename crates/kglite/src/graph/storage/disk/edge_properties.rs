// Disk-backed edge property storage. Replaces the heap-only
// `HashMap<u32, Vec<(InternedKey, Value)>>` that blew RAM at Wikidata
// scale (30–60 GB).
//
// Layout: per-edge columnar slots indexed by edge_idx.
//
//   edge_prop_offsets.bin  MmapOrVec<u64>, (max_edge_idx + 1) entries.
//                          offsets[i]..offsets[i+1] = byte range in heap
//                          for edge i's Postcard-serialized props blob.
//                          offsets[i] == offsets[i+1] means "no props".
//   edge_prop_heap.bin     MmapBytes, variable-length. Each populated
//                          slot holds a Postcard `Vec<(u64, Value)>` —
//                          raw InternedKey hashes, no interner needed
//                          on the read path.
//
// Runtime state: columnar `base` (read-only mmap) + HashMap `overlay`
// for edges mutated since last save. The overlay grows with mutation
// count, not graph size — disk mode's bounded-memory rule forbids heap
// structures that scale with the graph.
//
// Sequential access pattern: iterating edges in edge_idx order reads
// offsets and heap linearly. Random single-edge lookups incur one
// page fault per file, which is unavoidable and still cheaper than
// the retired zstd-decode-whole-HashMap-at-load path.
//
// Pre-0.14 format 0/1 graphs are rejected before payload decoding.

use crate::datatypes::values::Value;
use crate::graph::schema::{InternedKey, StringInterner};
use crate::graph::storage::mapped::mmap_vec::{MmapBytes, MmapOrVec};
use std::borrow::Cow;
use std::collections::HashMap;
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Columnar base filenames, written into the graph's data directory
/// alongside the CSR/column files.
pub const OFFSETS_FILE: &str = "edge_prop_offsets.bin";
pub const HEAP_FILE: &str = "edge_prop_heap.bin";

/// Retired pre-0.14 combined edge-property file, removed on current saves.
pub const LEGACY_FILE: &str = "edge_properties.bin.zst";

/// Lengths needed to mmap the columnar files at load time. Persisted
/// in `DiskGraphMeta` so the loader can `load_mapped` without scanning
/// file sizes at runtime.
#[derive(Debug, serde::Serialize, serde::Deserialize, Default, Clone, Copy)]
pub struct EdgePropertyStoreMeta {
    /// Number of u64 offsets written. Equal to `(upper_bound + 1)` passed
    /// to `save_to`. `offsets_len * 8` is the byte size of the offsets file.
    pub offsets_len: usize,
    /// Heap byte length. The mmap'd file may be padded but only
    /// `heap_len` bytes are valid.
    pub heap_len: usize,
}

/// Disk-backed columnar snapshot of edge properties. Read-only after load.
#[derive(Debug)]
struct ColumnarBase {
    /// offsets[edge_idx]..offsets[edge_idx + 1] = byte range in heap.
    offsets: MmapOrVec<u64>,
    /// Concatenated Postcard blobs, one `Vec<(u64, Value)>` per populated slot.
    heap: MmapBytes,
    codec: crate::serde_codec::CodecVersion,
}

impl ColumnarBase {
    /// Byte slice for a single edge's props blob, or empty slice if
    /// the edge has no properties in this base.
    fn slot(&self, edge_idx: u32) -> Option<&[u8]> {
        let i = edge_idx as usize;
        // offsets has length (upper_bound + 1). Edges past that are
        // "not in base" — caller must consult overlay.
        if i + 1 >= self.offsets.len() {
            return None;
        }
        let start = self.offsets.get(i) as usize;
        let end = self.offsets.get(i + 1) as usize;
        if start == end {
            // Empty slot (sparsity encoding).
            return Some(&[]);
        }
        Some(self.heap.slice(start, end))
    }

    /// Upper bound on edge_idx values covered by this base (exclusive).
    fn len(&self) -> u32 {
        // (upper_bound + 1) offsets written; the final trailing offset is
        // the total heap length.
        self.offsets.len().saturating_sub(1) as u32
    }
}

/// Decode a single slot's bytes into `(InternedKey, Value)` pairs.
/// Assumes bytes were produced by `encode_props_into` at save time.
fn decode_props(
    codec: crate::serde_codec::CodecVersion,
    bytes: &[u8],
) -> Option<Vec<(InternedKey, Value)>> {
    let raw: Vec<(u64, Value)> = crate::serde_codec::decode_exact_with(
        codec,
        bytes,
        bytes.len() as u64,
        crate::serde_codec::DecodeLimits::new(bytes.len() as u64, bytes.len() as u64),
    )
    .ok()?;
    Some(
        raw.into_iter()
            .map(|(k, v)| (InternedKey::from_u64(k), v))
            .collect(),
    )
}

/// `path` with a `.tmp` suffix — the staging name used when a save has to
/// replace files this store is currently mapping.
fn staged_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".tmp");
    PathBuf::from(name)
}

/// Encode `(InternedKey, Value)` pairs directly into the provided heap
/// buffer. Avoids the per-edge `Vec<u8>` allocation that dominated measured
/// save-path cost. The interner is not consulted — we store the raw u64 hash.
fn encode_props_into(props: &[(InternedKey, Value)], heap: &mut Vec<u8>) -> io::Result<()> {
    let raw: Vec<(u64, &Value)> = props.iter().map(|(k, v)| (k.as_u64(), v)).collect();
    let encoded =
        crate::serde_codec::encode_versioned(crate::serde_codec::CURRENT_CODEC, &raw, u64::MAX)
            .map_err(io::Error::other)?;
    heap.extend_from_slice(&encoded);
    Ok(())
}

/// Edge-property store: columnar disk base + in-memory mutation overlay.
///
/// `None` overlay entries are tombstones (edge was deleted or its props
/// were explicitly emptied). `Some(vec)` entries replace whatever the
/// base had for that edge. Entries absent from the overlay fall through
/// to the base.
#[derive(Debug, Default)]
pub struct EdgePropertyStore {
    base: Option<Arc<ColumnarBase>>,
    overlay: HashMap<u32, Option<Vec<(InternedKey, Value)>>>,
}

impl EdgePropertyStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Lookup an edge's current properties.
    /// - Overlay hit: returns `Cow::Borrowed` (zero copy).
    /// - Overlay tombstone: returns `None`.
    /// - Base hit: deserializes the columnar slot into an owned `Vec`
    ///   and wraps in `Cow::Owned`.
    pub fn get(&self, edge_idx: u32) -> Option<Cow<'_, [(InternedKey, Value)]>> {
        // Overlay first. `Some(None)` = explicit tombstone, hide base.
        if let Some(entry) = self.overlay.get(&edge_idx) {
            return entry.as_ref().map(|v| Cow::Borrowed(v.as_slice()));
        }
        let base = self.base.as_ref()?;
        let bytes = base.slot(edge_idx)?;
        if bytes.is_empty() {
            return None;
        }
        let decoded = decode_props(base.codec, bytes)?;
        if decoded.is_empty() {
            None
        } else {
            Some(Cow::Owned(decoded))
        }
    }

    /// Replace an edge's properties in the overlay.
    pub fn insert(&mut self, edge_idx: u32, props: Vec<(InternedKey, Value)>) {
        if props.is_empty() {
            // Normalise to tombstone/absent so `is_empty()` is meaningful.
            self.remove(edge_idx);
            return;
        }
        self.overlay.insert(edge_idx, Some(props));
    }

    /// Remove an edge's properties. Writes a tombstone into the overlay if
    /// the base might contain this edge; otherwise just drops the overlay entry.
    pub fn remove(&mut self, edge_idx: u32) {
        if let Some(base) = self.base.as_ref() {
            if edge_idx < base.len() {
                self.overlay.insert(edge_idx, None);
                return;
            }
        }
        self.overlay.remove(&edge_idx);
    }

    /// Remove and return the current properties, if any. Mirrors
    /// `HashMap::remove`. Used by `compact_edges` which remaps edge_idx.
    pub fn take(&mut self, edge_idx: u32) -> Option<Vec<(InternedKey, Value)>> {
        let current = self
            .get(edge_idx)
            .map(|cow| cow.into_owned())
            .filter(|v| !v.is_empty());
        self.remove(edge_idx);
        current
    }

    /// True when no edge currently has any properties. Conservative: a
    /// loaded graph with all entries tombstoned in the overlay still
    /// reports `false`. That's OK — the save path always writes a valid
    /// (possibly all-zero-length) file and callers gate on the existence
    /// of at least one property-bearing edge.
    pub fn is_empty(&self) -> bool {
        if self.base.as_ref().is_some_and(|b| b.len() > 0) {
            return false;
        }
        self.overlay.values().all(|v| v.is_none())
    }

    /// Smallest exclusive upper bound on edge_idx values that *might*
    /// have properties in this store. Used by callers (like `compact_edges`)
    /// that need to iterate every potentially-populated slot to remap
    /// indices.
    pub fn upper_bound(&self) -> u32 {
        let base_upper = self.base.as_ref().map(|b| b.len()).unwrap_or(0);
        let overlay_upper = self
            .overlay
            .keys()
            .max()
            .copied()
            .map(|k| k.saturating_add(1))
            .unwrap_or(0);
        base_upper.max(overlay_upper)
    }

    /// Number of edges whose properties are held in the heap mutation
    /// overlay rather than the mmap-backed columnar base. Tombstones
    /// (`None` entries) count: they occupy a heap slot too.
    ///
    /// Observability only — surfaced as `graph_info()['edge_property_overlay_rows']`
    /// so a caller can see how much of a disk graph's edge-property data is
    /// still resident rather than paged.
    pub(crate) fn overlay_len(&self) -> usize {
        self.overlay.len()
    }

    /// Fork a transaction overlay while sharing the immutable columnar base.
    /// Only the mutation-sized overlay is copied.
    pub(crate) fn fork_overlay(&self) -> Self {
        Self {
            base: self.base.clone(),
            overlay: self.overlay.clone(),
        }
    }

    /// Write the current merged state (base ∪ overlay, minus tombstones)
    /// to `target_dir` as the columnar format, then clear the overlay.
    ///
    /// `upper_bound` is an exclusive upper bound on edge_idx values that
    /// need slots in the offsets array. Typically `DiskGraph::next_edge_idx`.
    /// Slots past any actual data are represented as zero-length entries.
    ///
    /// After `save_to` returns, `self.base` is *not* automatically
    /// re-opened — callers that want subsequent reads to hit the freshly
    /// written files should `*self = Self::load_from(target_dir, 2, meta, ...)`.
    pub fn save_to(&mut self, target_dir: &Path, upper_bound: u32) -> io::Result<()> {
        let offsets_path = target_dir.join(OFFSETS_FILE);
        let heap_path = target_dir.join(HEAP_FILE);

        // Is this store's own columnar base the thing we are about to
        // overwrite? Then the new files are staged beside the old ones and
        // swapped in at the end: the sweep below reads *through* the base, and
        // overwriting a mapped file in place is UB per memmap2's docs.
        //
        // Releasing the mapping up front — what this used to do — satisfied
        // the UB rule by throwing the data away: every edge whose properties
        // lived only in the base then swept as absent, and the save wrote an
        // empty store over a full one. Unreachable from `DirGraph::save_disk`,
        // which always writes a fresh generation stage, but `save_to_dir` and
        // `seal_to_new_segment` both target the live data dir directly.
        let writing_in_place =
            self.base.as_ref().and_then(|b| b.offsets.file_path()) == Some(offsets_path.as_path());

        // Fast path: skip the O(upper_bound) sweep when no edge has any
        // properties. The sweep ran 6.7M overlay HashMap lookups and wrote a
        // 54 MB all-zero offsets file on every save of a property-less
        // wiki500m graph — ~1–2 s per save. Zero-length files instead; reload
        // resolves them to an empty base (matching guard in `load_from`) and
        // every `get(edge_idx)` returns `None`, as the full sweep did.
        if self.is_empty() {
            // A base mapping these paths contributes nothing readable here
            // (`is_empty` already proved it holds no edges), so releasing it
            // before the truncating write loses no data and keeps the write
            // off a live mapping.
            if writing_in_place {
                self.base = None;
            }
            std::fs::write(&offsets_path, b"")?;
            std::fs::write(&heap_path, b"")?;
            let legacy = target_dir.join(LEGACY_FILE);
            if legacy.exists() {
                let _ = std::fs::remove_file(&legacy);
            }
            self.overlay.clear();
            return Ok(());
        }

        let mut offsets: Vec<u64> = Vec::with_capacity(upper_bound as usize + 1);
        let heap_hint: usize = self
            .overlay
            .values()
            .map(|v| v.as_ref().map_or(0, |p| 32 + 16 * p.len()))
            .sum();
        let mut heap: Vec<u8> = Vec::with_capacity(heap_hint);

        for edge_idx in 0..upper_bound {
            offsets.push(heap.len() as u64);
            if let Some(cow) = self.get(edge_idx) {
                if !cow.is_empty() {
                    encode_props_into(cow.as_ref(), &mut heap)?;
                }
            }
        }
        offsets.push(heap.len() as u64);

        let (offsets_out, heap_out) = if writing_in_place {
            (staged_path(&offsets_path), staged_path(&heap_path))
        } else {
            (offsets_path.clone(), heap_path.clone())
        };
        MmapOrVec::from_vec(offsets).save_to_file(&offsets_out)?;
        std::fs::write(&heap_out, &heap)?;
        if writing_in_place {
            // Release the mapping only now that everything it held has been
            // written elsewhere. POSIX keeps the mapping valid across the
            // rename; Windows refuses to replace a file with an open handle.
            self.base = None;
            std::fs::rename(&offsets_out, &offsets_path)?;
            std::fs::rename(&heap_out, &heap_path)?;
        }

        // Drop the retired pre-0.14 combined file if a rebuilt graph is being
        // written over a directory that still carries one.
        let legacy = target_dir.join(LEGACY_FILE);
        if legacy.exists() {
            let _ = std::fs::remove_file(&legacy);
        }

        // Overlay has been fully absorbed into the new base file.
        self.overlay.clear();
        Ok(())
    }

    /// Open the store from a directory.
    /// - `format_version` comes from `DiskGraphMeta.edge_properties_format`
    ///   (2 = Postcard columnar).
    /// - `meta` provides the file lengths needed to mmap the columnar files.
    /// - `_interner` is retained by the storage boundary; current columnar
    ///   payloads store raw u64 hashes and never touch it.
    pub fn load_from(
        dir: &Path,
        format_version: u8,
        meta: EdgePropertyStoreMeta,
        _interner: &mut StringInterner,
    ) -> io::Result<Self> {
        if format_version < 2 {
            return Err(crate::graph::io::file::pre_014_bincode_error(
                "edge-property store",
            ));
        }
        if format_version > 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported edge property format {format_version}"),
            ));
        }
        Self::open_base(dir, meta)
    }

    /// Map an already-validated format-2 columnar base out of `dir`.
    ///
    /// Split out of [`Self::load_from`] so the streaming writer can hand back
    /// a live store without inventing a `StringInterner` the read path never
    /// consults.
    fn open_base(dir: &Path, meta: EdgePropertyStoreMeta) -> io::Result<Self> {
        let offsets_path = dir.join(OFFSETS_FILE);
        let heap_path = dir.join(HEAP_FILE);
        if !offsets_path.exists() {
            return Ok(Self::new());
        }
        // The save path emits zero-length offsets/heap files for the "no
        // properties anywhere" case, and `MmapOrVec::load_mapped` with
        // `len == 0` would `map_mut` a zero-byte region, which fails on some
        // platforms. Short-circuit to an empty store: `get()` resolves via the
        // overlay check (base is `None`) and returns `None`, which is the
        // semantic of "no edge has props".
        if meta.offsets_len == 0 {
            return Ok(Self::new());
        }
        let offsets = MmapOrVec::<u64>::load_mapped(&offsets_path, meta.offsets_len)?;
        let heap = MmapBytes::load_mapped(&heap_path, meta.heap_len)?;
        Ok(Self {
            base: Some(Arc::new(ColumnarBase {
                offsets,
                heap,
                codec: crate::serde_codec::CodecVersion::PostcardV1,
            })),
            overlay: HashMap::new(),
        })
    }

    /// On-disk metadata for the columnar files in `dir` — call after `save_to`
    /// or [`EdgePropertyWriter::finish`] to get the values that belong in
    /// `DiskGraphMeta`. The counts are read back off the files, so they reflect
    /// what was written, not any in-memory state.
    pub fn meta_for(dir: &Path) -> EdgePropertyStoreMeta {
        let offsets = dir.join(OFFSETS_FILE);
        let heap = dir.join(HEAP_FILE);
        EdgePropertyStoreMeta {
            offsets_len: std::fs::metadata(&offsets)
                .map(|m| m.len() as usize / std::mem::size_of::<u64>())
                .unwrap_or(0),
            heap_len: std::fs::metadata(&heap)
                .map(|m| m.len() as usize)
                .unwrap_or(0),
        }
    }
}

/// Streaming builder for a columnar base, written one edge at a time.
///
/// [`EdgePropertyStore::save_to`] persists the *merged* state, which means
/// everything it writes has to be reachable through `get()` first — a bulk
/// materializer would have to assemble the whole `HashMap` overlay before it
/// could save, which is ~175 B per property-bearing edge of heap holding data
/// that is about to be written and never read from the heap again. That was
/// the dominant term in `enable_disk_mode()`'s memory growth.
///
/// This writer appends straight into `edge_prop_offsets.bin` /
/// `edge_prop_heap.bin` in ascending `edge_idx` order, so the live heap is one
/// edge's encoded blob. The bytes it emits are byte-identical to `save_to`'s
/// (same sparsity encoding, same `encode_props_into` framing, same
/// `edge_properties_format = 2`), and [`Self::finish`] hands back a store
/// mapping what it wrote — the same end state a reload reaches.
pub struct EdgePropertyWriter {
    dir: PathBuf,
    offsets: io::BufWriter<std::fs::File>,
    heap: io::BufWriter<std::fs::File>,
    /// Number of slots whose offset has been written — i.e. the next
    /// `edge_idx` this writer expects.
    slots: u32,
    heap_len: u64,
    scratch: Vec<u8>,
    wrote_any: bool,
}

impl EdgePropertyWriter {
    /// Open a writer over `dir`, truncating any existing columnar files.
    ///
    /// `dir` must be the directory the resulting graph will treat as its data
    /// dir (`seg_000/`), so the blob lands beside the CSR arrays that index
    /// into it and `DiskGraphMeta` describes both from one place.
    pub fn create(dir: &Path) -> io::Result<Self> {
        Ok(Self {
            dir: dir.to_path_buf(),
            offsets: io::BufWriter::new(std::fs::File::create(dir.join(OFFSETS_FILE))?),
            heap: io::BufWriter::with_capacity(
                1 << 16,
                std::fs::File::create(dir.join(HEAP_FILE))?,
            ),
            slots: 0,
            heap_len: 0,
            scratch: Vec::new(),
            wrote_any: false,
        })
    }

    /// Record `edge_idx`'s properties. Edges must arrive in ascending
    /// `edge_idx` order; every index skipped gets the zero-length slot that
    /// encodes "no properties", exactly as `save_to`'s sweep would write it.
    ///
    /// Empty `props` are ignored rather than stored, matching
    /// [`EdgePropertyStore::insert`]'s normalisation.
    pub fn push(&mut self, edge_idx: u32, props: &[(InternedKey, Value)]) -> io::Result<()> {
        debug_assert!(
            edge_idx >= self.slots,
            "EdgePropertyWriter requires ascending edge_idx: {edge_idx} after {}",
            self.slots
        );
        if props.is_empty() {
            return Ok(());
        }
        self.pad_to(edge_idx)?;
        self.offsets.write_all(&self.heap_len.to_ne_bytes())?;
        self.scratch.clear();
        encode_props_into(props, &mut self.scratch)?;
        self.heap.write_all(&self.scratch)?;
        self.heap_len += self.scratch.len() as u64;
        self.slots += 1;
        self.wrote_any = true;
        Ok(())
    }

    fn pad_to(&mut self, edge_idx: u32) -> io::Result<()> {
        while self.slots < edge_idx {
            self.offsets.write_all(&self.heap_len.to_ne_bytes())?;
            self.slots += 1;
        }
        Ok(())
    }

    /// Close the files and map them back as a read-only base.
    ///
    /// `upper_bound` is exclusive and matches `save_to`'s: `(upper_bound + 1)`
    /// offsets are written, the last being the total heap length. When no edge
    /// carried properties the two files are emitted zero-length and the store
    /// comes back with no base at all — the same representation
    /// `save_to`'s empty fast path produces, and the one `load_from`
    /// recognises via `offsets_len == 0`.
    pub fn finish(mut self, upper_bound: u32) -> io::Result<EdgePropertyStore> {
        debug_assert!(
            upper_bound >= self.slots,
            "upper_bound {upper_bound} is below the {} slots already written",
            self.slots
        );
        if !self.wrote_any {
            drop(self.offsets);
            drop(self.heap);
            std::fs::write(self.dir.join(OFFSETS_FILE), b"")?;
            std::fs::write(self.dir.join(HEAP_FILE), b"")?;
            return Ok(EdgePropertyStore::new());
        }
        self.pad_to(upper_bound)?;
        self.offsets.write_all(&self.heap_len.to_ne_bytes())?;
        // `into_inner` flushes; the files are then closed by dropping them.
        // No fsync — `save_to` does not fsync either, and the durability
        // boundary for a disk graph is the generation publish, not this write.
        drop(self.offsets.into_inner().map_err(|e| e.into_error())?);
        drop(self.heap.into_inner().map_err(|e| e.into_error())?);

        let meta = EdgePropertyStore::meta_for(&self.dir);
        EdgePropertyStore::open_base(&self.dir, meta)
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;
    use crate::datatypes::values::Value;
    use crate::graph::schema::StringInterner;
    use tempfile::TempDir;

    fn k(s: &str, interner: &mut StringInterner) -> InternedKey {
        interner.get_or_intern(s)
    }

    #[test]
    fn empty_store_is_empty() {
        let s = EdgePropertyStore::new();
        assert!(s.is_empty());
    }

    #[test]
    fn insert_and_get_overlay_hit() {
        let mut interner = StringInterner::new();
        let mut s = EdgePropertyStore::new();
        let props = vec![(k("weight", &mut interner), Value::Float64(1.5))];
        s.insert(42, props.clone());
        let got = s.get(42).expect("should hit overlay");
        assert_eq!(got.as_ref(), props.as_slice());
        assert!(!s.is_empty());
    }

    #[test]
    fn remove_without_base_drops_entry() {
        let mut interner = StringInterner::new();
        let mut s = EdgePropertyStore::new();
        s.insert(7, vec![(k("x", &mut interner), Value::Int64(1))]);
        s.remove(7);
        assert!(s.get(7).is_none());
        assert!(s.is_empty());
    }

    #[test]
    fn insert_empty_normalises_to_absent() {
        let mut s = EdgePropertyStore::new();
        s.insert(1, vec![]);
        assert!(s.get(1).is_none());
        assert!(s.is_empty());
    }

    #[test]
    fn save_and_load_round_trip() {
        let tmp = TempDir::new().unwrap();
        let mut interner = StringInterner::new();
        let mut s = EdgePropertyStore::new();

        let p0 = vec![
            (k("name", &mut interner), Value::String("alpha".into())),
            (k("rank", &mut interner), Value::Int64(7)),
        ];
        let p1 = vec![(k("weight", &mut interner), Value::Float64(3.14))];
        s.insert(0, p0.clone());
        s.insert(3, p1.clone()); // edges 1, 2 have no props

        s.save_to(tmp.path(), 4).unwrap();

        let meta = EdgePropertyStore::meta_for(tmp.path());
        // 4 slots + 1 trailing offset = 5 offsets
        assert_eq!(meta.offsets_len, 5);
        assert!(meta.heap_len > 0);

        let reloaded = EdgePropertyStore::load_from(tmp.path(), 2, meta, &mut interner).unwrap();
        assert_eq!(reloaded.get(0).unwrap().as_ref(), p0.as_slice());
        assert!(reloaded.get(1).is_none());
        assert!(reloaded.get(2).is_none());
        assert_eq!(reloaded.get(3).unwrap().as_ref(), p1.as_slice());
    }

    #[test]
    fn overlay_tombstones_hide_base() {
        let tmp = TempDir::new().unwrap();
        let tmp2 = TempDir::new().unwrap();
        let mut interner = StringInterner::new();

        let mut s = EdgePropertyStore::new();
        s.insert(5, vec![(k("a", &mut interner), Value::Int64(99))]);
        s.save_to(tmp.path(), 6).unwrap();

        let meta = EdgePropertyStore::meta_for(tmp.path());
        let mut reloaded =
            EdgePropertyStore::load_from(tmp.path(), 2, meta, &mut interner).unwrap();
        assert!(reloaded.get(5).is_some());

        reloaded.remove(5);
        assert!(reloaded.get(5).is_none());

        // Save+reload persists the tombstone — no trace of edge 5.
        reloaded.save_to(tmp2.path(), 6).unwrap();
        let meta2 = EdgePropertyStore::meta_for(tmp2.path());
        let after = EdgePropertyStore::load_from(tmp2.path(), 2, meta2, &mut interner).unwrap();
        assert!(after.get(5).is_none());
    }

    #[test]
    fn take_returns_and_removes() {
        let mut interner = StringInterner::new();
        let mut s = EdgePropertyStore::new();
        let p = vec![(k("t", &mut interner), Value::Boolean(true))];
        s.insert(11, p.clone());
        let taken = s.take(11).unwrap();
        assert_eq!(taken, p);
        assert!(s.get(11).is_none());
    }

    #[test]
    fn pre_014_combined_store_is_rejected() {
        let tmp = TempDir::new().unwrap();
        let mut interner = StringInterner::new();
        let error = EdgePropertyStore::load_from(
            tmp.path(),
            0,
            EdgePropertyStoreMeta::default(),
            &mut interner,
        )
        .unwrap_err();
        assert!(error.to_string().contains("pre-0.14"));
    }

    #[test]
    fn pre_014_columnar_and_unknown_versions_fail() {
        let tmp = TempDir::new().unwrap();
        let mut interner = StringInterner::new();
        let key = k("legacy-column", &mut interner);
        let raw = vec![(key.as_u64(), Value::Int64(17))];
        let heap =
            crate::serde_codec::encode_versioned(crate::serde_codec::CURRENT_CODEC, &raw, u64::MAX)
                .unwrap();
        MmapOrVec::from_vec(vec![0u64, heap.len() as u64])
            .save_to_file(&tmp.path().join(OFFSETS_FILE))
            .unwrap();
        std::fs::write(tmp.path().join(HEAP_FILE), &heap).unwrap();
        let meta = EdgePropertyStore::meta_for(tmp.path());

        let error = EdgePropertyStore::load_from(tmp.path(), 1, meta, &mut interner).unwrap_err();
        assert!(error.to_string().contains("pre-0.14"));

        let error = EdgePropertyStore::load_from(tmp.path(), 99, meta, &mut interner).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error
            .to_string()
            .contains("unsupported edge property format"));
    }

    #[test]
    fn writer_streams_the_same_bytes_save_to_would_have_written() {
        // The streaming path is only safe if it is byte-identical to the
        // sweep it replaces — the loader reads both through the same
        // `edge_properties_format = 2` framing.
        let streamed = TempDir::new().unwrap();
        let swept = TempDir::new().unwrap();
        let mut interner = StringInterner::new();

        let p0 = vec![
            (k("name", &mut interner), Value::String("alpha".into())),
            (k("rank", &mut interner), Value::Int64(7)),
        ];
        let p3 = vec![(k("weight", &mut interner), Value::Float64(3.14))];

        let mut writer = EdgePropertyWriter::create(streamed.path()).unwrap();
        writer.push(0, &p0).unwrap();
        writer.push(1, &[]).unwrap(); // empty props are not stored
        writer.push(3, &p3).unwrap();
        let store = writer.finish(6).unwrap();

        let mut reference = EdgePropertyStore::new();
        reference.insert(0, p0.clone());
        reference.insert(3, p3.clone());
        reference.save_to(swept.path(), 6).unwrap();

        for name in [OFFSETS_FILE, HEAP_FILE] {
            assert_eq!(
                std::fs::read(streamed.path().join(name)).unwrap(),
                std::fs::read(swept.path().join(name)).unwrap(),
                "{name} diverges from the sweep's bytes"
            );
        }

        // And the returned store already maps what it wrote — no reload,
        // and nothing left on the heap.
        assert_eq!(store.overlay_len(), 0);
        assert_eq!(store.get(0).unwrap().as_ref(), p0.as_slice());
        assert!(store.get(1).is_none());
        assert!(store.get(2).is_none());
        assert_eq!(store.get(3).unwrap().as_ref(), p3.as_slice());
        assert!(store.get(5).is_none());
        assert!(!store.is_empty());
        assert_eq!(store.upper_bound(), 6);
    }

    #[test]
    fn writer_with_no_properties_emits_the_empty_representation() {
        let tmp = TempDir::new().unwrap();
        let mut writer = EdgePropertyWriter::create(tmp.path()).unwrap();
        writer.push(0, &[]).unwrap();
        let store = writer.finish(1_000).unwrap();

        assert!(store.is_empty());
        assert!(store.get(0).is_none());
        let meta = EdgePropertyStore::meta_for(tmp.path());
        assert_eq!(meta.offsets_len, 0, "no sweep-sized all-zero offsets file");
        assert_eq!(meta.heap_len, 0);
    }

    #[test]
    fn writer_output_takes_overlay_writes_on_top() {
        // A converted graph is mutated after conversion (`SET r.p`): the new
        // value has to land in the overlay and win over the streamed base.
        let tmp = TempDir::new().unwrap();
        let mut interner = StringInterner::new();
        let key = k("weight", &mut interner);

        let mut writer = EdgePropertyWriter::create(tmp.path()).unwrap();
        writer.push(1, &[(key, Value::Int64(1))]).unwrap();
        let mut store = writer.finish(3).unwrap();

        store.insert(1, vec![(key, Value::Int64(42))]);
        assert_eq!(store.get(1).unwrap().as_ref()[0].1, Value::Int64(42));
        store.insert(2, vec![(key, Value::Int64(7))]);
        assert_eq!(store.get(2).unwrap().as_ref()[0].1, Value::Int64(7));

        // A save merges base and overlay; the base survives being read from
        // while its own files are replaced.
        store.save_to(tmp.path(), 3).unwrap();
        let meta = EdgePropertyStore::meta_for(tmp.path());
        let reloaded = EdgePropertyStore::load_from(tmp.path(), 2, meta, &mut interner).unwrap();
        assert_eq!(reloaded.get(1).unwrap().as_ref()[0].1, Value::Int64(42));
        assert_eq!(reloaded.get(2).unwrap().as_ref()[0].1, Value::Int64(7));
    }

    #[test]
    fn save_to_over_its_own_mapped_base_keeps_the_base_rows() {
        // `save_to` writes the *merged* state, which it reads through
        // `self.base`. Releasing that mapping before the sweep — to avoid
        // writing into a mapped file — silently wrote an empty store over a
        // full one whenever the target was the base's own directory
        // (`save_to_dir` / `seal_to_new_segment` both do exactly that).
        let tmp = TempDir::new().unwrap();
        let mut interner = StringInterner::new();
        let based = vec![(k("a", &mut interner), Value::Int64(1))];
        let added = vec![(k("b", &mut interner), Value::Int64(2))];

        let mut store = EdgePropertyStore::new();
        store.insert(0, based.clone());
        store.save_to(tmp.path(), 2).unwrap();
        let meta = EdgePropertyStore::meta_for(tmp.path());
        let mut store = EdgePropertyStore::load_from(tmp.path(), 2, meta, &mut interner).unwrap();
        store.insert(1, added.clone());

        store.save_to(tmp.path(), 2).unwrap();

        let meta = EdgePropertyStore::meta_for(tmp.path());
        let reloaded = EdgePropertyStore::load_from(tmp.path(), 2, meta, &mut interner).unwrap();
        assert_eq!(
            reloaded.get(0).map(|c| c.into_owned()),
            Some(based),
            "the base's own row was dropped by an in-place save"
        );
        assert_eq!(reloaded.get(1).map(|c| c.into_owned()), Some(added));
        assert!(
            !tmp.path().join(format!("{OFFSETS_FILE}.tmp")).exists(),
            "staging file left behind"
        );
    }

    #[test]
    fn empty_store_save_emits_zero_length_files_and_reload_preserves_semantics() {
        // Pins the empty-store fast path: when no edge has properties, save
        // must not sweep `0..upper_bound` or write a 54 MB all-zero offsets
        // file. Verify the files are zero-length *and* that a reload with a
        // large `upper_bound` answers `get()` as None for every edge.
        let tmp = TempDir::new().unwrap();
        let mut interner = StringInterner::new();
        let mut s = EdgePropertyStore::new();

        // upper_bound is deliberately large — the sweep would have written
        // 1_000_000 * 8 = 8 MB of zeros.
        s.save_to(tmp.path(), 1_000_000).unwrap();

        let offsets_meta = std::fs::metadata(tmp.path().join(OFFSETS_FILE)).unwrap();
        let heap_meta = std::fs::metadata(tmp.path().join(HEAP_FILE)).unwrap();
        assert_eq!(offsets_meta.len(), 0, "offsets file should be empty");
        assert_eq!(heap_meta.len(), 0, "heap file should be empty");

        let meta = EdgePropertyStore::meta_for(tmp.path());
        assert_eq!(meta.offsets_len, 0);
        assert_eq!(meta.heap_len, 0);

        let reloaded = EdgePropertyStore::load_from(tmp.path(), 2, meta, &mut interner).unwrap();
        assert!(reloaded.is_empty());
        // get() on any edge_idx returns None — matches the sweep's behavior
        // where every edge had an empty slot.
        assert!(reloaded.get(0).is_none());
        assert!(reloaded.get(999_999).is_none());
        assert!(reloaded.get(u32::MAX).is_none());
    }
}
