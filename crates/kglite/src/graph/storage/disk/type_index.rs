//! Mmap-resident `type_indices.bin` store with overlay for mutations.
//!
//! Replaces the eager `zstd::decode_all` + 124M-`Vec::push` rebuild. Reads
//! come from a memory-mapped flat CSR; mutations land in an in-memory
//! overlay that takes precedence over the base. On save, overlay + base
//! are merged into a fresh `.bin`.
//!
//! ## File format `type_indices.bin`
//!
//! ```text
//! Header (32 bytes):
//!   [ 0.. 8]  magic        = b"KGLTIDXR"  (R = raw, mmap-friendly)
//!   [ 8..12]  version      = u32 LE (= 1)
//!   [12..16]  num_types    = u32 LE
//!   [16..24]  total_nodes  = u64 LE
//!   [24..32]  data_offset  = u64 LE   (32 + 24 * num_types)
//!
//! Directory at [32]: 24 bytes per entry, sorted by type_key:
//!   [ 0.. 8]  type_key:    u64 LE  (InternedKey)
//!   [ 8..16]  payload_off: u64 LE   (file-relative)
//!   [16..24]  payload_len: u64 LE   (= 4 * num_entries for that type)
//!
//! Data section: contiguous `[u32]` slices per type (NodeIndex values).
//! ```
//!
//! Lookup is `O(log num_types)` directory probe at load (cached as a
//! `HashMap<String, BaseEntry>`) plus `O(1)` slice access.

use memmap2::Mmap;
use petgraph::graph::NodeIndex;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::graph::schema::{InternedKey, StringInterner};

const MAGIC: &[u8; 8] = b"KGLTIDXR";
const VERSION: u32 = 1;
const HEADER_BYTES: usize = 32;
const DIR_ENTRY_BYTES: usize = 24;

fn invalid_index(message: &str) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("invalid type_indices.bin: {message}"),
    )
}

fn read_le_u32(bytes: &[u8], index: usize) -> Option<u32> {
    let start = index.checked_mul(4)?;
    Some(u32::from_le_bytes(
        bytes.get(start..start.checked_add(4)?)?.try_into().ok()?,
    ))
}

fn le_u32_iter(bytes: &[u8]) -> impl Iterator<Item = u32> + '_ {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
}

fn le_u32_binary_search(bytes: &[u8], wanted: u32) -> bool {
    let mut low = 0usize;
    let mut high = bytes.len() / 4;
    while low < high {
        let mid = low + (high - low) / 2;
        match read_le_u32(bytes, mid).unwrap().cmp(&wanted) {
            std::cmp::Ordering::Less => low = mid + 1,
            std::cmp::Ordering::Greater => high = mid,
            std::cmp::Ordering::Equal => return true,
        }
    }
    false
}

/// Mmap-backed read-only view of `type_indices.bin`.
#[derive(Debug)]
pub struct TypeIndexBase {
    mmap: Arc<Mmap>,
    /// type_name -> (file-relative offset, num_entries). Built once at load.
    dir: HashMap<String, BaseEntry>,
}

#[derive(Debug, Clone, Copy)]
struct BaseEntry {
    payload_off: u64,
    num_entries: u32,
}

impl TypeIndexBase {
    /// Load `type_indices.bin` from `dir`. Returns `Ok(None)` if absent or magic mismatch.
    pub fn load_from(dir: &Path, interner: &StringInterner) -> std::io::Result<Option<Self>> {
        let path = dir.join("type_indices.bin");
        if !path.exists() {
            return Ok(None);
        }
        let file = std::fs::File::open(&path)?;
        let len = file.metadata()?.len() as usize;
        if len < HEADER_BYTES {
            return Ok(None);
        }
        // SAFETY: opened above; KGLite holds the GIL during load and no
        // other process writes to the file concurrently.
        let mmap = unsafe { Mmap::map(&file)? };
        if &mmap[..8] != MAGIC {
            return Ok(None);
        }
        let version = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
        if version != VERSION {
            return Err(invalid_index("unsupported raw index version"));
        }
        let num_types = u32::from_le_bytes(mmap[12..16].try_into().unwrap()) as usize;
        let declared_total = u64::from_le_bytes(mmap[16..24].try_into().unwrap());
        let data_offset = usize::try_from(u64::from_le_bytes(mmap[24..32].try_into().unwrap()))
            .map_err(|_| invalid_index("data offset exceeds usize"))?;
        let dir_bytes = DIR_ENTRY_BYTES
            .checked_mul(num_types)
            .ok_or_else(|| invalid_index("directory size overflow"))?;
        let need = HEADER_BYTES
            .checked_add(dir_bytes)
            .ok_or_else(|| invalid_index("directory offset overflow"))?;
        if len < need || data_offset != need {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "type_indices.bin has an invalid directory/data boundary",
            ));
        }

        let mut dir_map: HashMap<String, BaseEntry> = HashMap::with_capacity(num_types);
        let mut previous_key = None;
        let mut expected_payload = data_offset;
        let mut total_entries = 0u64;
        for i in 0..num_types {
            let off = HEADER_BYTES + i * DIR_ENTRY_BYTES;
            let type_key = u64::from_le_bytes(mmap[off..off + 8].try_into().unwrap());
            let payload_off = u64::from_le_bytes(mmap[off + 8..off + 16].try_into().unwrap());
            let payload_len = u64::from_le_bytes(mmap[off + 16..off + 24].try_into().unwrap());
            if previous_key.is_some_and(|previous| type_key <= previous) {
                return Err(invalid_index("directory keys are not strictly increasing"));
            }
            previous_key = Some(type_key);
            if payload_len % 4 != 0 {
                return Err(invalid_index("payload length is not divisible by four"));
            }
            let payload_off_usize = usize::try_from(payload_off)
                .map_err(|_| invalid_index("payload offset exceeds usize"))?;
            let payload_len_usize = usize::try_from(payload_len)
                .map_err(|_| invalid_index("payload length exceeds usize"))?;
            let payload_end = payload_off_usize
                .checked_add(payload_len_usize)
                .ok_or_else(|| invalid_index("payload range overflow"))?;
            if payload_off_usize != expected_payload || payload_end > len {
                return Err(invalid_index(
                    "payloads overlap, contain gaps, or exceed the file",
                ));
            }
            let mut previous_node = None;
            for node in le_u32_iter(&mmap[payload_off_usize..payload_end]) {
                if previous_node.is_some_and(|previous| node <= previous) {
                    return Err(invalid_index("node indices are not strictly increasing"));
                }
                previous_node = Some(node);
            }
            expected_payload = payload_end;
            let num_entries = payload_len / 4;
            total_entries = total_entries
                .checked_add(num_entries)
                .ok_or_else(|| invalid_index("entry count overflow"))?;
            let num_entries = u32::try_from(num_entries)
                .map_err(|_| invalid_index("one type contains too many entries"))?;
            let name = interner
                .try_resolve(InternedKey::from_u64(type_key))
                .ok_or_else(|| invalid_index("directory contains an unresolved type key"))?;
            if dir_map
                .insert(
                    name.to_string(),
                    BaseEntry {
                        payload_off,
                        num_entries,
                    },
                )
                .is_some()
            {
                return Err(invalid_index("duplicate resolved type name"));
            }
        }
        if expected_payload != len || total_entries != declared_total {
            return Err(invalid_index(
                "payload cardinality does not match the header",
            ));
        }

        Ok(Some(Self {
            mmap: Arc::new(mmap),
            dir: dir_map,
        }))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.dir.contains_key(name)
    }

    /// Slice of u32 NodeIndex values for `name`, mapped directly from the file.
    pub fn slice_for(&self, name: &str) -> Option<&[u8]> {
        let entry = self.dir.get(name)?;
        let n = entry.num_entries as usize;
        let off = entry.payload_off as usize;
        if n == 0 {
            return Some(&[]);
        }
        self.mmap.get(off..off.checked_add(n.checked_mul(4)?)?)
    }

    /// Materialize a base entry into an owned Vec. Used on save and on
    /// first mutation when the entry must be promoted into the overlay.
    pub fn materialize(&self, name: &str) -> Option<Vec<NodeIndex>> {
        let slice = self.slice_for(name)?;
        Some(
            le_u32_iter(slice)
                .map(|u| NodeIndex::new(u as usize))
                .collect(),
        )
    }
}

/// View into either the overlay's `Vec<NodeIndex>` or canonical little-endian
/// bytes borrowed directly from the mmap.
pub enum TypeNodesRef<'a> {
    Overlay(&'a [NodeIndex]),
    Mmap(&'a [u8]),
}

impl<'a> TypeNodesRef<'a> {
    pub fn len(&self) -> usize {
        match self {
            TypeNodesRef::Overlay(s) => s.len(),
            TypeNodesRef::Mmap(s) => s.len() / 4,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> TypeNodesIter<'_> {
        match self {
            TypeNodesRef::Overlay(s) => TypeNodesIter::Overlay(s.iter()),
            TypeNodesRef::Mmap(s) => TypeNodesIter::Mmap(s.chunks_exact(4)),
        }
    }

    pub fn to_vec(&self) -> Vec<NodeIndex> {
        match self {
            TypeNodesRef::Overlay(s) => s.to_vec(),
            TypeNodesRef::Mmap(s) => le_u32_iter(s).map(|u| NodeIndex::new(u as usize)).collect(),
        }
    }

    pub fn get(&self, i: usize) -> Option<NodeIndex> {
        match self {
            TypeNodesRef::Overlay(s) => s.get(i).copied(),
            TypeNodesRef::Mmap(s) => read_le_u32(s, i).map(|u| NodeIndex::new(u as usize)),
        }
    }

    /// Linear scan for membership. O(n); used in tests and light callers
    /// (delete paths use a HashSet built from the slice instead).
    #[allow(dead_code)]
    pub fn contains(&self, idx: &NodeIndex) -> bool {
        match self {
            TypeNodesRef::Overlay(s) => s.contains(idx),
            TypeNodesRef::Mmap(s) => le_u32_iter(s).any(|u| u as usize == idx.index()),
        }
    }

    /// O(log n) membership test. Relies on the sortedness invariant of
    /// `TypeIndexStore`: entries are inserted in `node_indices()` iteration
    /// order (0, 1, …, n-1), and filtering by type produces a naturally
    /// sorted subsequence. `write_type_indices_bin` preserves that order
    /// across save + reload.
    ///
    /// If a caller has mutated an `Overlay` slice with `push` (or via the
    /// `entry_or_default` path) without re-sorting, this method may give
    /// false negatives — see the `contains` fallback above.
    pub fn binary_search_idx(&self, idx: NodeIndex) -> bool {
        match self {
            TypeNodesRef::Overlay(s) => s.binary_search(&idx).is_ok(),
            TypeNodesRef::Mmap(s) => {
                let want = idx.index() as u32;
                le_u32_binary_search(s, want)
            }
        }
    }
}

pub enum TypeNodesIter<'a> {
    Overlay(std::slice::Iter<'a, NodeIndex>),
    Mmap(std::slice::ChunksExact<'a, u8>),
}

impl<'a> Iterator for TypeNodesIter<'a> {
    type Item = NodeIndex;
    #[inline]
    fn next(&mut self) -> Option<NodeIndex> {
        match self {
            TypeNodesIter::Overlay(it) => it.next().copied(),
            TypeNodesIter::Mmap(it) => it.next().map(|bytes| {
                NodeIndex::new(u32::from_le_bytes(bytes.try_into().unwrap()) as usize)
            }),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            TypeNodesIter::Overlay(it) => it.size_hint(),
            TypeNodesIter::Mmap(it) => it.size_hint(),
        }
    }
}

impl ExactSizeIterator for TypeNodesIter<'_> {
    fn len(&self) -> usize {
        match self {
            TypeNodesIter::Overlay(it) => it.len(),
            TypeNodesIter::Mmap(it) => it.len(),
        }
    }
}

/// HashMap-shaped wrapper around an optional mmap base + overlay.
#[derive(Default, Clone)]
pub struct TypeIndexStore {
    overlay: HashMap<String, Vec<NodeIndex>>,
    /// Types that exist in `base` but were removed/invalidated post-load.
    removed: std::collections::HashSet<String>,
    base: Option<Arc<TypeIndexBase>>,
}

impl TypeIndexStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_base(base: TypeIndexBase) -> Self {
        Self {
            overlay: HashMap::new(),
            removed: std::collections::HashSet::new(),
            base: Some(Arc::new(base)),
        }
    }

    pub fn contains_key(&self, name: &str) -> bool {
        if self.overlay.contains_key(name) {
            return true;
        }
        if self.removed.contains(name) {
            return false;
        }
        self.base.as_ref().is_some_and(|b| b.contains(name))
    }

    pub fn get(&self, name: &str) -> Option<TypeNodesRef<'_>> {
        if let Some(v) = self.overlay.get(name) {
            return Some(TypeNodesRef::Overlay(v.as_slice()));
        }
        if self.removed.contains(name) {
            return None;
        }
        let base = self.base.as_deref()?;
        base.slice_for(name).map(TypeNodesRef::Mmap)
    }

    pub fn remove(&mut self, name: &str) -> Option<Vec<NodeIndex>> {
        let prev = self.overlay.remove(name);
        if self.base.as_ref().is_some_and(|b| b.contains(name)) {
            self.removed.insert(name.to_string());
        }
        prev
    }

    pub fn clear(&mut self) {
        self.overlay.clear();
        if let Some(base) = &self.base {
            self.removed.extend(base.dir.keys().cloned());
        }
    }

    pub fn len(&self) -> usize {
        let base_count = self
            .base
            .as_ref()
            .map(|b| b.dir.keys().filter(|k| !self.removed.contains(*k)).count())
            .unwrap_or(0);
        let overlay_only = self
            .overlay
            .keys()
            .filter(|k| self.base.as_ref().map(|b| !b.contains(k)).unwrap_or(true))
            .count();
        base_count + overlay_only
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate live type names (overlay first, then base entries that aren't
    /// shadowed by overlay or marked removed).
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        let overlay_names: Vec<&str> = self.overlay.keys().map(|s| s.as_str()).collect();
        let base_names: Vec<&str> = self
            .base
            .as_ref()
            .map(|b| {
                b.dir
                    .keys()
                    .filter(|k| {
                        !self.overlay.contains_key(k.as_str()) && !self.removed.contains(k.as_str())
                    })
                    .map(|s| s.as_str())
                    .collect()
            })
            .unwrap_or_default();
        overlay_names.into_iter().chain(base_names)
    }

    /// Iterate `(name, TypeNodesRef)` for every live entry.
    pub fn iter(&self) -> impl Iterator<Item = (&str, TypeNodesRef<'_>)> {
        let overlay_pairs: Vec<(&str, TypeNodesRef<'_>)> = self
            .overlay
            .iter()
            .map(|(k, v)| (k.as_str(), TypeNodesRef::Overlay(v.as_slice())))
            .collect();
        let base_pairs: Vec<(&str, TypeNodesRef<'_>)> = match self.base.as_deref() {
            Some(base) => base
                .dir
                .iter()
                .filter(|(k, _)| {
                    !self.overlay.contains_key(k.as_str()) && !self.removed.contains(k.as_str())
                })
                .filter_map(|(k, _)| {
                    base.slice_for(k.as_str())
                        .map(|s| (k.as_str(), TypeNodesRef::Mmap(s)))
                })
                .collect(),
            None => Vec::new(),
        };
        overlay_pairs.into_iter().chain(base_pairs)
    }

    /// HashMap-`entry`-shaped accessor: materialize any base entry into the
    /// overlay before returning a mutable Vec reference (or insert empty).
    pub fn entry_or_default(&mut self, name: String) -> &mut Vec<NodeIndex> {
        if !self.overlay.contains_key(&name) && !self.removed.contains(&name) {
            if let Some(base) = self.base.as_deref() {
                if let Some(v) = base.materialize(&name) {
                    self.overlay.insert(name.clone(), v);
                }
            }
        }
        self.removed.remove(&name);
        self.overlay.entry(name).or_default()
    }

    /// Promote a single type into the overlay if needed, then run `predicate`
    /// on its Vec via `Vec::retain`. No-op if the type is absent.
    pub fn retain_in_type<F: FnMut(&NodeIndex) -> bool>(&mut self, name: &str, predicate: F) {
        if let Some(v) = self.overlay.get_mut(name) {
            v.retain(predicate);
            return;
        }
        if self.removed.contains(name) {
            return;
        }
        // Materialize base into overlay then retain.
        if let Some(base) = self.base.as_deref() {
            if let Some(mut v) = base.materialize(name) {
                v.retain(predicate);
                self.overlay.insert(name.to_string(), v);
            }
        }
    }

    /// Run `predicate.retain(...)` across every live Vec. Materializes every
    /// base entry into the overlay first — used by full-graph rebuild paths.
    pub fn retain_all<F: FnMut(&NodeIndex) -> bool + Copy>(&mut self, predicate: F) {
        // Materialize all base entries into the overlay.
        if let Some(base) = self.base.clone() {
            for name in base.dir.keys() {
                if !self.overlay.contains_key(name.as_str())
                    && !self.removed.contains(name.as_str())
                {
                    if let Some(v) = base.materialize(name) {
                        self.overlay.insert(name.clone(), v);
                    }
                }
            }
            // After full materialization, drop the base reference so subsequent
            // reads come exclusively from the overlay.
            self.base = None;
            self.removed.clear();
        }
        for v in self.overlay.values_mut() {
            v.retain(predicate);
        }
    }

    /// Replace the entire store with a fresh HashMap.
    pub fn replace_with(&mut self, map: HashMap<String, Vec<NodeIndex>>) {
        self.overlay = map;
        self.removed.clear();
        self.base = None;
    }
}

// =============================================================================
// Writer
// =============================================================================

pub fn write_type_indices_bin(
    dir: &Path,
    store: &TypeIndexStore,
    interner: &StringInterner,
) -> Result<(), String> {
    // Collect (type_key, name, slice-or-vec) sorted by type_key.
    enum Source<'a> {
        Slice(&'a [u8]),
        Vec(&'a [NodeIndex]),
    }
    impl Source<'_> {
        fn len(&self) -> usize {
            match self {
                Source::Slice(s) => s.len() / 4,
                Source::Vec(s) => s.len(),
            }
        }
        fn write_into(&self, out: &mut Vec<u8>) {
            match self {
                Source::Slice(s) => out.extend_from_slice(s),
                Source::Vec(s) => {
                    for n in s.iter() {
                        out.extend_from_slice(&(n.index() as u32).to_le_bytes());
                    }
                }
            }
        }
    }

    let mut interner_clone = interner.clone();
    let mut entries: Vec<(u64, Source<'_>)> = Vec::new();
    for (name, view) in store.iter() {
        let key = interner_clone
            .try_get_or_intern(name)
            .map_err(|e| e.to_string())?
            .as_u64();
        let src = match view {
            TypeNodesRef::Overlay(s) => Source::Vec(s),
            TypeNodesRef::Mmap(s) => Source::Slice(s),
        };
        entries.push((key, src));
    }
    entries.sort_by_key(|(k, _)| *k);

    let num_types = entries.len();
    let total_nodes: u64 = entries.iter().map(|(_, s)| s.len() as u64).sum();
    let header_size = HEADER_BYTES;
    let dir_size = DIR_ENTRY_BYTES * num_types;
    let data_offset = (header_size + dir_size) as u64;

    // Pre-compute per-type payload offsets.
    let mut offsets: Vec<(u64, u64)> = Vec::with_capacity(num_types);
    let mut cursor = data_offset;
    for (_, src) in &entries {
        let len = src.len() as u64 * 4;
        offsets.push((cursor, len));
        cursor += len;
    }

    let total = cursor as usize;
    let mut out = Vec::with_capacity(total);
    // Header
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(num_types as u32).to_le_bytes());
    out.extend_from_slice(&total_nodes.to_le_bytes());
    out.extend_from_slice(&data_offset.to_le_bytes());

    // Directory
    for ((type_key, _), (off, len)) in entries.iter().zip(offsets.iter()) {
        out.extend_from_slice(&type_key.to_le_bytes());
        out.extend_from_slice(&off.to_le_bytes());
        out.extend_from_slice(&len.to_le_bytes());
    }

    // Data section
    for (_, src) in &entries {
        src.write_into(&mut out);
    }

    debug_assert_eq!(out.len(), total);

    std::fs::write(dir.join("type_indices.bin"), out)
        .map_err(|e| format!("Failed to write type_indices.bin: {}", e))?;
    Ok(())
}

#[cfg(test)]
mod validation_tests {
    use super::*;

    fn fixture(type_key: u64, nodes: &[u32]) -> Vec<u8> {
        let data_offset = HEADER_BYTES + DIR_ENTRY_BYTES;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&(nodes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(data_offset as u64).to_le_bytes());
        bytes.extend_from_slice(&type_key.to_le_bytes());
        bytes.extend_from_slice(&(data_offset as u64).to_le_bytes());
        bytes.extend_from_slice(&((nodes.len() * 4) as u64).to_le_bytes());
        for node in nodes {
            bytes.extend_from_slice(&node.to_le_bytes());
        }
        bytes
    }

    fn load(bytes: &[u8], interner: &StringInterner) -> std::io::Result<Option<TypeIndexBase>> {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("type_indices.bin"), bytes).unwrap();
        TypeIndexBase::load_from(temp.path(), interner)
    }

    #[test]
    fn valid_little_endian_fixture_round_trips() {
        let mut interner = StringInterner::new();
        let key = interner.get_or_intern("Person").as_u64();
        let base = load(&fixture(key, &[1, 7, 42]), &interner)
            .unwrap()
            .unwrap();
        assert_eq!(
            base.materialize("Person").unwrap(),
            vec![NodeIndex::new(1), NodeIndex::new(7), NodeIndex::new(42)]
        );
    }

    #[test]
    fn rejects_directory_arithmetic_and_payload_shape_errors() {
        let mut interner = StringInterner::new();
        let key = interner.get_or_intern("Person").as_u64();
        let valid = fixture(key, &[1, 7]);

        let mut directory_overflow = valid.clone();
        directory_overflow[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(
            load(&directory_overflow, &interner).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );

        let mut bad_boundary = valid.clone();
        bad_boundary[24..32].copy_from_slice(&33u64.to_le_bytes());
        assert_eq!(
            load(&bad_boundary, &interner).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );

        let mut indivisible = valid.clone();
        indivisible[48..56].copy_from_slice(&7u64.to_le_bytes());
        assert_eq!(
            load(&indivisible, &interner).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );

        let mut past_eof = valid.clone();
        past_eof[40..48].copy_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(
            load(&past_eof, &interner).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn rejects_header_cardinality_and_unsorted_nodes() {
        let mut interner = StringInterner::new();
        let key = interner.get_or_intern("Person").as_u64();
        let mut wrong_total = fixture(key, &[1, 7]);
        wrong_total[16..24].copy_from_slice(&3u64.to_le_bytes());
        assert_eq!(
            load(&wrong_total, &interner).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );

        let unsorted = fixture(key, &[7, 1]);
        assert_eq!(
            load(&unsorted, &interner).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        let duplicate = fixture(key, &[7, 7]);
        assert_eq!(
            load(&duplicate, &interner).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
    }
}
