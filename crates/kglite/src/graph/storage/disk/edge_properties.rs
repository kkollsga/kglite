// src/graph/storage/disk/edge_properties.rs
//
// Disk-backed edge property storage. Replaces the heap-only
// `HashMap<u32, Vec<(InternedKey, Value)>>` that blew RAM at Wikidata
// scale (30–60 GB). PR2 of the disk-graph-improvement-plan.
//
// Layout: per-edge columnar slots indexed by edge_idx.
//
//   edge_prop_offsets.bin  MmapOrVec<u64>, (max_edge_idx + 1) entries.
//                          offsets[i]..offsets[i+1] = byte range in heap
//                          for edge i's bincode-serialized props blob.
//                          offsets[i] == offsets[i+1] means "no props".
//   edge_prop_heap.bin     MmapBytes, variable-length. Each populated
//                          slot holds a bincode `Vec<(u64, Value)>` —
//                          raw InternedKey hashes, no interner needed
//                          on the read path.
//
// Runtime state: columnar `base` (read-only mmap) + HashMap `overlay`
// for edges mutated since last save. The overlay grows with mutation
// count, not graph size — bounded by workload, per the bounded-memory
// constraint in `feedback_bounded_memory.md`.
//
// Sequential access pattern: iterating edges in edge_idx order reads
// offsets and heap linearly. Random single-edge lookups incur one
// page fault per file, which is unavoidable and still cheaper than
// the current zstd-decode-whole-HashMap-at-load path.
//
// Legacy format=0 graphs stored `Vec<(InternedKey, Value)>` under the
// `SerdeSerializeGuard` (InternedKey serialized as strings). The
// `load_legacy` path still handles these for backward compat and
// requires `&mut StringInterner` to re-register keys. New format=1
// reads never touch the interner.

use crate::datatypes::values::Value;
use crate::graph::schema::{InternedKey, SerdeDeserializeGuard, StringInterner};
use crate::graph::storage::mapped::mmap_vec::{MmapBytes, MmapOrVec};
use std::borrow::Cow;
use std::collections::HashMap;
use std::io;
use std::path::Path;

/// Filename constants for the columnar base. Both live under the graph's
/// data directory alongside the existing CSR/column files.
pub const OFFSETS_FILE: &str = "edge_prop_offsets.bin";
pub const HEAP_FILE: &str = "edge_prop_heap.bin";

/// Legacy combined bincode+zstd blob emitted by format=0 graphs.
/// Kept readable via `load_legacy` for backward compat.
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
    /// Concatenated bincode blobs. Each populated slot is a
    /// `Vec<(u64, Value)>` — u64 is the raw `InternedKey` hash.
    heap: MmapBytes,
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
        // (upper_bound + 1) offsets written; final trailing offset is
        // the total heap length. So the number of edges covered is
        // offsets.len().saturating_sub(1).
        self.offsets.len().saturating_sub(1) as u32
    }
}

/// Decode a single slot's bytes into `(InternedKey, Value)` pairs.
/// Assumes bytes were produced by `encode_props_into` at save time.
fn decode_props(bytes: &[u8]) -> Option<Vec<(InternedKey, Value)>> {
    let raw: Vec<(u64, Value)> = bincode::deserialize(bytes).ok()?;
    Some(
        raw.into_iter()
            .map(|(k, v)| (InternedKey::from_u64(k), v))
            .collect(),
    )
}

/// Encode `(InternedKey, Value)` pairs directly into the provided heap
/// buffer. Avoids the per-edge `Vec<u8>` allocation that dominated
/// save-path cost during PR2 phase-2 benchmarking. The interner is not
/// consulted — we store the raw u64 hash.
fn encode_props_into(props: &[(InternedKey, Value)], heap: &mut Vec<u8>) -> io::Result<()> {
    // bincode's default config encodes Vec<T> as a u64 length prefix +
    // N × T-encoded. We mirror that layout by serializing a
    // Vec<(u64, &Value)> — but by writing directly into `heap` via
    // `serialize_into`, we save one Vec<u8> allocation per edge (1M+
    // allocations on a million-edge save).
    let raw: Vec<(u64, &Value)> = props.iter().map(|(k, v)| (k.as_u64(), v)).collect();
    bincode::serialize_into(heap, &raw).map_err(io::Error::other)
}

/// Edge-property store: columnar disk base + in-memory mutation overlay.
///
/// `None` overlay entries are tombstones (edge was deleted or its props
/// were explicitly emptied). `Some(vec)` entries replace whatever the
/// base had for that edge. Entries absent from the overlay fall through
/// to the base.
#[derive(Debug, Default)]
pub struct EdgePropertyStore {
    base: Option<ColumnarBase>,
    overlay: HashMap<u32, Option<Vec<(InternedKey, Value)>>>,
}

impl EdgePropertyStore {
    /// Empty store — no base, empty overlay. Used by fresh in-memory builds.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct from a pre-populated HashMap. Used when migrating a
    /// legacy format=0 graph — the caller bincode-deserializes the old
    /// blob and passes it here.
    pub fn from_overlay(map: HashMap<u32, Vec<(InternedKey, Value)>>) -> Self {
        Self {
            base: None,
            overlay: map.into_iter().map(|(k, v)| (k, Some(v))).collect(),
        }
    }

    /// Lookup an edge's current properties.
    /// - Overlay hit: returns `Cow::Borrowed` (zero copy).
    /// - Overlay tombstone: returns `None`.
    /// - Base hit: deserializes the columnar slot into an owned `Vec`
    ///   and wraps in `Cow::Owned`. No interner needed — the blob stores
    ///   raw `InternedKey` u64 hashes directly.
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
        let decoded = decode_props(bytes)?;
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
    /// indices. The returned value is `max(base.len(), max_overlay_idx + 1)`.
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

    /// Deep clone — for `shallow_copy()` paths in DiskGraph that produce
    /// a fresh in-memory graph from a disk-backed one.
    ///
    /// Base files are mmap'd and cannot be trivially cloned at runtime,
    /// so we materialize everything currently visible into the new
    /// store's overlay. This uses more RAM temporarily but preserves
    /// semantics; if `shallow_copy` grows hot on disk graphs, revisit
    /// and consider hard-linking or mmap sharing.
    pub fn deep_clone(&self) -> Self {
        let mut new = EdgePropertyStore::new();
        if let Some(base) = self.base.as_ref() {
            for edge_idx in 0..base.len() {
                if let Some(cow) = self.get(edge_idx) {
                    new.insert(edge_idx, cow.into_owned());
                }
            }
        }
        for (idx, entry) in &self.overlay {
            if let Some(props) = entry {
                new.insert(*idx, props.clone());
            }
        }
        new
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
    /// written files should `*self = Self::load_from(target_dir, 1, meta, ...)`.
    pub fn save_to(&mut self, target_dir: &Path, upper_bound: u32) -> io::Result<()> {
        let offsets_path = target_dir.join(OFFSETS_FILE);
        let heap_path = target_dir.join(HEAP_FILE);

        // Release any existing mmap over these exact paths before we
        // overwrite them — memmap2 docs are explicit that overlapping
        // writes to a mapped file are UB.
        if let Some(base) = self.base.as_ref() {
            if base.offsets.file_path() == Some(&offsets_path) {
                self.base = None;
            }
        }

        // 0.8.12 phase-1 fast path: skip the O(upper_bound) sweep when
        // no edge has any properties. The pre-phase-1 loop ran 6.7M
        // overlay HashMap lookups and wrote a 54 MB all-zero offsets
        // file on every save of a property-less wiki500m graph — ~1–2 s
        // per save. Now we emit zero-length files and reload resolves
        // to an empty base (see matching guard in `load_from`). Every
        // `get(edge_idx)` on the reloaded store returns `None`, which
        // is exactly what the full-sweep path produced anyway.
        if self.is_empty() {
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
        // Pre-size heap based on overlay content — avoids repeated realloc.
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

        MmapOrVec::from_vec(offsets).save_to_file(&offsets_path)?;
        std::fs::write(&heap_path, &heap)?;

        // Remove the legacy file if it's lingering from a format=0 load.
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
    ///   (0 = legacy, 1 = columnar).
    /// - `meta` provides the file lengths needed to mmap the columnar files.
    ///   Ignored when loading format=0.
    /// - `interner` is mutated only on the legacy path (to register keys
    ///   deserialized from the old string-keyed format). The columnar path
    ///   stores raw u64 hashes and never touches the interner.
    pub fn load_from(
        dir: &Path,
        format_version: u8,
        meta: EdgePropertyStoreMeta,
        interner: &mut StringInterner,
    ) -> io::Result<Self> {
        if format_version == 0 || (meta.offsets_len == 0 && !dir.join(OFFSETS_FILE).exists()) {
            return Self::load_legacy(dir, interner);
        }
        let offsets_path = dir.join(OFFSETS_FILE);
        let heap_path = dir.join(HEAP_FILE);
        if !offsets_path.exists() {
            return Ok(Self::new());
        }
        // 0.8.12 phase-1: the save path now emits zero-length
        // offsets/heap files for the "no properties anywhere" case.
        // `MmapOrVec::load_mapped` with `len == 0` would try to
        // `map_mut` a zero-byte region, which fails on some platforms.
        // Short-circuit to an empty store — every `get()` on this
        // store will resolve via the overlay check (base is None) and
        // return `None`, matching the semantic of "no edge has props".
        if meta.offsets_len == 0 {
            return Ok(Self::new());
        }
        let offsets = MmapOrVec::<u64>::load_mapped(&offsets_path, meta.offsets_len)?;
        let heap = MmapBytes::load_mapped(&heap_path, meta.heap_len)?;
        Ok(Self {
            base: Some(ColumnarBase { offsets, heap }),
            overlay: HashMap::new(),
        })
    }

    fn load_legacy(dir: &Path, interner: &mut StringInterner) -> io::Result<Self> {
        let legacy = dir.join(LEGACY_FILE);
        if !legacy.exists() {
            return Ok(Self::new());
        }
        let compressed = std::fs::read(&legacy)?;
        let bytes = zstd::decode_all(compressed.as_slice()).map_err(io::Error::other)?;
        let _guard = SerdeDeserializeGuard::new(interner);
        let map: HashMap<u32, Vec<(InternedKey, Value)>> =
            bincode::deserialize(&bytes).map_err(io::Error::other)?;
        Ok(Self::from_overlay(map))
    }

    /// Compute the on-disk metadata for this store — call after `save_to`
    /// to get the values that belong in `DiskGraphMeta`. The returned
    /// counts reflect what `save_to` just wrote, not the in-memory state.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatypes::values::Value;
    use crate::graph::schema::{SerdeSerializeGuard, StringInterner};
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

        // Reload and verify each edge.
        let reloaded = EdgePropertyStore::load_from(tmp.path(), 1, meta, &mut interner).unwrap();
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
            EdgePropertyStore::load_from(tmp.path(), 1, meta, &mut interner).unwrap();
        assert!(reloaded.get(5).is_some());

        reloaded.remove(5);
        assert!(reloaded.get(5).is_none());

        // Save+reload persists the tombstone — no trace of edge 5.
        reloaded.save_to(tmp2.path(), 6).unwrap();
        let meta2 = EdgePropertyStore::meta_for(tmp2.path());
        let after = EdgePropertyStore::load_from(tmp2.path(), 1, meta2, &mut interner).unwrap();
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
    fn legacy_load_reads_hashmap_blob() {
        let tmp = TempDir::new().unwrap();
        let mut interner = StringInterner::new();
        let mut map: HashMap<u32, Vec<(InternedKey, Value)>> = HashMap::new();
        map.insert(
            2,
            vec![(k("legacy", &mut interner), Value::String("old".into()))],
        );

        // Emit the legacy bincode+zstd blob manually, under the
        // serialization guard so InternedKey serializes as a string.
        {
            let _g = SerdeSerializeGuard::new(&interner);
            let raw = bincode::serialize(&map).unwrap();
            let compressed = zstd::encode_all(raw.as_slice(), 3).unwrap();
            std::fs::write(tmp.path().join(LEGACY_FILE), compressed).unwrap();
        }

        let loaded = EdgePropertyStore::load_from(
            tmp.path(),
            0,
            EdgePropertyStoreMeta::default(),
            &mut interner,
        )
        .unwrap();
        let got = loaded.get(2).expect("should load from legacy");
        assert_eq!(got.as_ref().len(), 1);
        assert_eq!(got.as_ref()[0].1, Value::String("old".into()));
    }

    #[test]
    fn empty_store_save_emits_zero_length_files_and_reload_preserves_semantics() {
        // 0.8.12 phase-1 fast path: when no edge has properties, save
        // must not sweep `0..upper_bound` or write a 54 MB all-zero
        // offsets file. Verify the files are zero-length *and* that a
        // reload with a large `upper_bound` answers `get()` as None for
        // every edge.
        let tmp = TempDir::new().unwrap();
        let mut interner = StringInterner::new();
        let mut s = EdgePropertyStore::new();

        // upper_bound is deliberately large — pre-phase-1 this would
        // have written 1_000_000 * 8 = 8 MB of zeros.
        s.save_to(tmp.path(), 1_000_000).unwrap();

        let offsets_meta = std::fs::metadata(tmp.path().join(OFFSETS_FILE)).unwrap();
        let heap_meta = std::fs::metadata(tmp.path().join(HEAP_FILE)).unwrap();
        assert_eq!(offsets_meta.len(), 0, "offsets file should be empty");
        assert_eq!(heap_meta.len(), 0, "heap file should be empty");

        let meta = EdgePropertyStore::meta_for(tmp.path());
        assert_eq!(meta.offsets_len, 0);
        assert_eq!(meta.heap_len, 0);

        let reloaded = EdgePropertyStore::load_from(tmp.path(), 1, meta, &mut interner).unwrap();
        assert!(reloaded.is_empty());
        // get() on any edge_idx returns None — matches the pre-phase-1
        // behavior where every edge had an empty slot.
        assert!(reloaded.get(0).is_none());
        assert!(reloaded.get(999_999).is_none());
        assert!(reloaded.get(u32::MAX).is_none());
    }
}
