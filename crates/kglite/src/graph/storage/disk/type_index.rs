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

use super::type_index_layer::TypeBucket;
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
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| u32::from_le_bytes(*chunk))
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
        // SAFETY: GraphDirectoryLock serializes disk-graph writers, which
        // publish a new immutable generation instead of truncating the
        // generation selected by this reader. This inode therefore remains
        // stable for the mapping's lifetime.
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

/// View into the overlay's members, or canonical little-endian bytes borrowed
/// directly from the mmap.
///
/// `Overlay` is what an unlayered bucket hands out — a plain slice, exactly as
/// before this field was layered — so the steady-state read path is unchanged.
/// `Layered` appears only while a fork is outstanding and is the concatenation
/// of its levels, in order (`disk/type_index_layer.rs`).
pub enum TypeNodesRef<'a> {
    Overlay(&'a [NodeIndex]),
    Mmap(&'a [u8]),
    Layered(&'a [Arc<Vec<NodeIndex>>]),
}

impl<'a> TypeNodesRef<'a> {
    pub fn len(&self) -> usize {
        match self {
            TypeNodesRef::Overlay(s) => s.len(),
            TypeNodesRef::Mmap(s) => s.len() / 4,
            TypeNodesRef::Layered(levels) => levels.iter().map(|level| level.len()).sum(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> TypeNodesIter<'_> {
        match self {
            TypeNodesRef::Overlay(s) => TypeNodesIter::Overlay(s.iter()),
            TypeNodesRef::Mmap(s) => TypeNodesIter::Mmap(s.as_chunks::<4>().0.iter()),
            TypeNodesRef::Layered(levels) => TypeNodesIter::Layered {
                levels,
                level: 0,
                pos: 0,
            },
        }
    }

    pub fn to_vec(&self) -> Vec<NodeIndex> {
        match self {
            TypeNodesRef::Overlay(s) => s.to_vec(),
            TypeNodesRef::Mmap(s) => le_u32_iter(s).map(|u| NodeIndex::new(u as usize)).collect(),
            TypeNodesRef::Layered(levels) => {
                let mut out = Vec::with_capacity(self.len());
                for level in levels.iter() {
                    out.extend_from_slice(level);
                }
                out
            }
        }
    }

    pub fn get(&self, i: usize) -> Option<NodeIndex> {
        match self {
            TypeNodesRef::Overlay(s) => s.get(i).copied(),
            TypeNodesRef::Mmap(s) => read_le_u32(s, i).map(|u| NodeIndex::new(u as usize)),
            TypeNodesRef::Layered(levels) => {
                let mut i = i;
                for level in levels.iter() {
                    if i < level.len() {
                        return Some(level[i]);
                    }
                    i -= level.len();
                }
                None
            }
        }
    }

    /// Linear scan for membership. O(n); used in tests and light callers
    /// (delete paths use a HashSet built from the slice instead).
    #[allow(dead_code)]
    pub fn contains(&self, idx: &NodeIndex) -> bool {
        match self {
            TypeNodesRef::Overlay(s) => s.contains(idx),
            TypeNodesRef::Mmap(s) => le_u32_iter(s).any(|u| u as usize == idx.index()),
            TypeNodesRef::Layered(levels) => levels.iter().any(|level| level.contains(idx)),
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
    ///
    /// **That is a live, if currently unreachable, wrong-answer risk, and the
    /// obvious repair is the wrong one.** A false negative here is not a slow
    /// lookup: the fused count-by-peer route uses this as a per-peer *type
    /// filter* (`match_clause/fused_match.rs`), so a member the search cannot
    /// find is a row dropped from a result. No failing executor input has been
    /// constructed — the fused paths that could produce one exclude via the
    /// generic route — but the invariant is an accident of `NodeIndex`
    /// allocation order, not something anything maintains.
    ///
    /// Sorting on insert (`TypeIndexStore::push_to_type`) was measured and
    /// **rejected**, 2026-08-15: it costs nothing on ingest (below the timer
    /// floor at 200k one-shot and 20k incremental creates), but it breaks a
    /// different contract. `DirGraph::appended_tail` defines a bulk append's
    /// delta as the bucket's *tail*, and both post-append index folds
    /// (`fold_appended_ids_into_index`, `fold_appended_into_user_indexes`) read
    /// it; an ordered insert puts a slot-reusing create in the middle, so the
    /// folds index the wrong nodes. `maintain::incremental_index_tests::
    /// deleting_then_recreating_an_id_repoints_the_index` fails outright on it
    /// — the recreated id ends up unindexed and a spurious duplicate-id warning
    /// fires. Insert order is load-bearing beyond this method.
    ///
    /// The shape that closes it without touching insert order: carry an
    /// `ascending` flag on `TypeBucket`, maintained O(1) in `push` (one
    /// comparison against the last member) and cleared by `to_mut`, the one
    /// unconstrained mutator; surface it through `TypeNodesRef`; fall back to
    /// the linear `contains` here when it is false.
    pub fn binary_search_idx(&self, idx: NodeIndex) -> bool {
        match self {
            TypeNodesRef::Overlay(s) => s.binary_search(&idx).is_ok(),
            TypeNodesRef::Mmap(s) => {
                let want = idx.index() as u32;
                le_u32_binary_search(s, want)
            }
            // Each level is a contiguous run of the sorted whole, so the
            // invariant this method documents holds per level; probe them all.
            TypeNodesRef::Layered(levels) => {
                levels.iter().any(|level| level.binary_search(&idx).is_ok())
            }
        }
    }
}

pub enum TypeNodesIter<'a> {
    Overlay(std::slice::Iter<'a, NodeIndex>),
    Mmap(std::slice::Iter<'a, [u8; 4]>),
    /// Walks the level stack in merge order. Only reachable while a fork is
    /// outstanding — an unlayered bucket hands out `Overlay`.
    ///
    /// The cursor is two `u32`s rather than two `usize`s, and the remaining
    /// count is derived rather than carried, so this variant stays one word
    /// wider than the two-pointer `Overlay`/`Mmap` arms. The label scan
    /// iterates this type in its hottest loop; growing it would tax every scan
    /// for a state only a forked graph can be in.
    Layered {
        levels: &'a [Arc<Vec<NodeIndex>>],
        level: u32,
        pos: u32,
    },
}

impl<'a> Iterator for TypeNodesIter<'a> {
    type Item = NodeIndex;
    #[inline]
    fn next(&mut self) -> Option<NodeIndex> {
        match self {
            TypeNodesIter::Overlay(it) => it.next().copied(),
            TypeNodesIter::Mmap(it) => it
                .next()
                .map(|bytes| NodeIndex::new(u32::from_le_bytes(*bytes) as usize)),
            TypeNodesIter::Layered { levels, level, pos } => {
                while (*level as usize) < levels.len() {
                    if let Some(idx) = levels[*level as usize].get(*pos as usize) {
                        *pos += 1;
                        return Some(*idx);
                    }
                    *level += 1;
                    *pos = 0;
                }
                None
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = ExactSizeIterator::len(self);
        (len, Some(len))
    }
}

impl ExactSizeIterator for TypeNodesIter<'_> {
    fn len(&self) -> usize {
        match self {
            TypeNodesIter::Overlay(it) => it.len(),
            TypeNodesIter::Mmap(it) => it.len(),
            TypeNodesIter::Layered { levels, level, pos } => levels
                .iter()
                .skip(*level as usize)
                .map(|entries| entries.len())
                .sum::<usize>()
                .saturating_sub(*pos as usize),
        }
    }
}

/// HashMap-shaped wrapper around an optional mmap base + overlay.
///
/// **The fork seam for `type_indices`.** Each overlay bucket is a stack of
/// shared, immutable levels ([`TypeBucket`]), so the derived `Clone` below
/// copies `Arc`s rather than the members themselves. Before that, this field
/// was the last O(V) term in `DirGraph::clone` on a plain graph.
#[derive(Default, Clone)]
pub struct TypeIndexStore {
    overlay: HashMap<String, TypeBucket>,
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

    /// Fold every bucket's level stack back into one, for the buckets this
    /// graph is the last holder of.
    ///
    /// Called at write entry alongside the backend's and `id_indices`'
    /// compaction, so "hold a view, write, drop the view, write again" returns
    /// to the flat representation on the next write. Per bucket this is an
    /// `Arc::get_mut` probe plus an O(delta) merge.
    pub fn try_compact(&mut self) {
        for bucket in self.overlay.values_mut() {
            bucket.try_compact();
        }
    }

    /// Append `idx` to `name`'s bucket without materialising it.
    ///
    /// **The `CREATE` path.** Unlike [`entry_or_default`](Self::entry_or_default)
    /// this never needs a mutable `Vec`, so a bucket shared with a fork grows a
    /// new level instead of copying a million entries — and it borrows the type
    /// name rather than taking it, so the common case (the bucket exists) also
    /// stops allocating a `String` per created node.
    pub fn push_to_type(&mut self, name: &str, idx: NodeIndex) {
        if let Some(bucket) = self.overlay.get_mut(name) {
            bucket.push(idx);
            return;
        }
        self.entry_or_default(name.to_string()).push(idx);
    }

    /// Reverse a journalled `BucketAppended` for `name`, editing this graph's
    /// own delta rather than a base a forked reader is holding.
    ///
    /// Falls back to the flattening retain when the entry is not in the
    /// writable tail — slower, still correct. See
    /// [`TypeBucket::undo_append`] for when that can happen.
    pub fn undo_append(&mut self, name: &str, idx: NodeIndex) {
        if let Some(bucket) = self.overlay.get_mut(name) {
            if bucket.undo_append(idx) {
                return;
            }
        }
        self.retain_in_type(name, |member| *member != idx);
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
        if let Some(bucket) = self.overlay.get(name) {
            return Some(match bucket.levels() {
                // The steady state: one level, handed out as the plain slice
                // this returned before the field was layered.
                [only] => TypeNodesRef::Overlay(only.as_slice()),
                levels => TypeNodesRef::Layered(levels),
            });
        }
        if self.removed.contains(name) {
            return None;
        }
        let base = self.base.as_deref()?;
        base.slice_for(name).map(TypeNodesRef::Mmap)
    }

    pub fn remove(&mut self, name: &str) -> Option<Vec<NodeIndex>> {
        let prev = self
            .overlay
            .remove(name)
            .map(|mut bucket| std::mem::take(bucket.to_mut()));
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
            .map(|(k, bucket)| {
                let view = match bucket.levels() {
                    [only] => TypeNodesRef::Overlay(only.as_slice()),
                    levels => TypeNodesRef::Layered(levels),
                };
                (k.as_str(), view)
            })
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
    ///
    /// A bucket shared with a fork is **flattened** here, because a `&mut Vec`
    /// cannot be served out of a stack of shared levels. Callers that only
    /// append should use [`push_to_type`](Self::push_to_type) instead, which
    /// stays O(1) while shared.
    pub fn entry_or_default(&mut self, name: String) -> &mut Vec<NodeIndex> {
        if !self.overlay.contains_key(&name) && !self.removed.contains(&name) {
            if let Some(base) = self.base.as_deref() {
                if let Some(v) = base.materialize(&name) {
                    self.overlay.insert(name.clone(), TypeBucket::from(v));
                }
            }
        }
        self.removed.remove(&name);
        self.overlay.entry(name).or_default().to_mut()
    }

    /// Locate `members` in `name`'s bucket, ascending by position.
    ///
    /// **The delete path's fast lane.** A `DETACH DELETE` used to reach this
    /// store only through [`retain_in_type`](Self::retain_in_type), which walks
    /// every member of the type and probes a `HashSet` for each one — 4.0 ms to
    /// remove a single node from a 1M-row bucket, and quadratic in a delete
    /// loop. Membership is resolvable in O(k log N) instead, via the
    /// sortedness invariant [`TypeNodesRef::binary_search_idx`] documents:
    /// members are appended in `node_indices()` order, so a type's bucket is a
    /// sorted subsequence.
    ///
    /// Returns `None` — meaning "use the retain" — whenever that invariant does
    /// not hold for one of the members. A freed `NodeIndex` slot reused by a
    /// later create is appended out of order, so the invariant is a fast path,
    /// never a guarantee. A position that *is* found is always right: the
    /// search only reports `Ok` for an element that compares equal.
    ///
    /// Promotes the bucket into a flat, owned overlay entry (as the retain
    /// does), which is what makes the returned positions valid coordinates for
    /// [`remove_positions`](Self::remove_positions) and for the statement
    /// journal's `BucketRemoved` entries.
    pub fn positions_of(
        &mut self,
        name: &str,
        members: &[NodeIndex],
    ) -> Option<Vec<(usize, NodeIndex)>> {
        if members.is_empty() {
            return Some(Vec::new());
        }
        if !self.contains_key(name) {
            return None;
        }
        let bucket = self.entry_or_default(name.to_string());
        let mut hits: Vec<(usize, NodeIndex)> = Vec::with_capacity(members.len());
        for member in members {
            hits.push((bucket.binary_search(member).ok()?, *member));
        }
        hits.sort_unstable();
        // Two doomed members resolving to one position means the bucket holds a
        // duplicate the search collapsed; the retain removes every occurrence,
        // so hand it back rather than removing one of them.
        let unique = hits.windows(2).all(|pair| pair[0].0 != pair[1].0);
        unique.then_some(hits)
    }

    /// Drop the members at `hits` (ascending positions, as returned by
    /// [`positions_of`](Self::positions_of)) from `name`'s bucket, **preserving
    /// the order of every survivor**.
    ///
    /// Order is not an aesthetic here: the bucket is the scan order of an
    /// un-`ORDER BY`'d `MATCH`, the save writer's row order, and the coordinate
    /// system the statement journal records its `BucketRemoved` positions in.
    /// So this closes the gaps with one `copy_within` per surviving run rather
    /// than swap-removing.
    pub fn remove_positions(&mut self, name: &str, hits: &[(usize, NodeIndex)]) {
        if hits.is_empty() {
            return;
        }
        let Some(bucket) = self.overlay.get_mut(name) else {
            return;
        };
        let bucket = bucket.to_mut();
        if hits.last().is_some_and(|(pos, _)| *pos >= bucket.len()) {
            debug_assert!(false, "stale bucket position handed to remove_positions");
            return;
        }
        let mut write = hits[0].0;
        for (i, (pos, _)) in hits.iter().enumerate() {
            let start = pos + 1;
            let end = hits
                .get(i + 1)
                .map(|(next, _)| *next)
                .unwrap_or(bucket.len());
            if end > start {
                bucket.copy_within(start..end, write);
                write += end - start;
            }
        }
        bucket.truncate(write);
    }

    /// Promote a single type into the overlay if needed, then run `predicate`
    /// on its Vec via `Vec::retain`. No-op if the type is absent.
    pub fn retain_in_type<F: FnMut(&NodeIndex) -> bool>(&mut self, name: &str, predicate: F) {
        if let Some(bucket) = self.overlay.get_mut(name) {
            bucket.to_mut().retain(predicate);
            return;
        }
        if self.removed.contains(name) {
            return;
        }
        // Materialize base into overlay then retain.
        if let Some(base) = self.base.as_deref() {
            if let Some(mut v) = base.materialize(name) {
                v.retain(predicate);
                self.overlay.insert(name.to_string(), TypeBucket::from(v));
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
                        self.overlay.insert(name.clone(), TypeBucket::from(v));
                    }
                }
            }
            // After full materialization, drop the base reference so subsequent
            // reads come exclusively from the overlay.
            self.base = None;
            self.removed.clear();
        }
        for bucket in self.overlay.values_mut() {
            bucket.to_mut().retain(predicate);
        }
    }

    /// Replace the entire store with a fresh HashMap.
    pub fn replace_with(&mut self, map: HashMap<String, Vec<NodeIndex>>) {
        self.overlay = map
            .into_iter()
            .map(|(name, members)| (name, TypeBucket::from(members)))
            .collect();
        self.removed.clear();
        self.base = None;
    }
}

// =============================================================================
// Writer
// =============================================================================

/// True when `members` ascends strictly. Early-exits on the first violation,
/// so the sorted case costs one linear compare pass and the unsorted case
/// stops as soon as it is decided.
fn strictly_increasing<I: Iterator<Item = u32>>(mut members: I) -> bool {
    let Some(mut previous) = members.next() else {
        return true;
    };
    for member in members {
        if member <= previous {
            return false;
        }
        previous = member;
    }
    true
}

/// Write `type_indices.bin` (flat CSR layout).
///
/// Type names are *resolved* against the interner that ships with the same
/// snapshot, never interned here — see the equivalent note on
/// [`write_id_indices_bin`](super::id_index::write_id_indices_bin). An
/// unregistered name has no persisted identity: an empty entry is dropped
/// (nothing to record), and a populated one fails the save rather than
/// shipping a directory whose type keys the loader cannot resolve.
///
/// **Each per-type payload is emitted in ascending `NodeIndex` order**, which
/// the in-memory bucket is not obliged to be in. A bucket is appended to in
/// creation order and a create reuses a slot a delete freed, so a
/// delete-then-create pair leaves the members out of order (`[0, 2, 1]` for the
/// three-node case). The reader validates every payload as strictly increasing
/// ([`TypeIndexBase::load_from`]) and `TypeNodesRef::binary_search_idx` binary-
/// searches the mmap arm with no linear fallback, so an unsorted payload is not
/// a file this crate can read: before this normalization the save *succeeded*
/// and the next load refused the graph outright — silent data loss.
///
/// Sorting here cannot desynchronize the columns. The payload is not a
/// coordinate system: a disk graph binds a node to its column row through
/// `DiskNodeSlot::row_id`, persisted per slot, and the portable `.kgl` path —
/// the one that *does* bind row k positionally
/// (`io/file/columns.rs::attach_portable_column_stores`) — never reads this
/// file, rebuilding `type_indices` from an ascending `node_indices()` scan
/// instead. The overlay positions the statement journal records
/// (`BucketRemoved`) are in-memory coordinates measured against the
/// materialized bucket and are never persisted.
pub fn write_type_indices_bin(
    dir: &Path,
    store: &TypeIndexStore,
    interner: &StringInterner,
) -> Result<(), String> {
    // Collect (type_key, slice-or-vec) sorted by type_key.
    enum Source<'a> {
        Slice(&'a [u8]),
        Vec(&'a [NodeIndex]),
        Levels(&'a [Arc<Vec<NodeIndex>>]),
        /// A payload the loop below had to reorder. Owned, because the
        /// borrowed forms are the graph's live buckets and a save must not
        /// mutate them.
        Sorted(Vec<u32>),
    }
    impl Source<'_> {
        fn len(&self) -> usize {
            match self {
                Source::Slice(s) => s.len() / 4,
                Source::Vec(s) => s.len(),
                Source::Levels(levels) => levels.iter().map(|level| level.len()).sum(),
                Source::Sorted(members) => members.len(),
            }
        }
        fn is_strictly_increasing(&self) -> bool {
            match self {
                Source::Slice(s) => strictly_increasing(le_u32_iter(s)),
                Source::Vec(s) => strictly_increasing(s.iter().map(|n| n.index() as u32)),
                Source::Levels(levels) => strictly_increasing(
                    levels
                        .iter()
                        .flat_map(|level| level.iter())
                        .map(|n| n.index() as u32),
                ),
                Source::Sorted(members) => strictly_increasing(members.iter().copied()),
            }
        }
        fn to_members(&self) -> Vec<u32> {
            match self {
                Source::Slice(s) => le_u32_iter(s).collect(),
                Source::Vec(s) => s.iter().map(|n| n.index() as u32).collect(),
                Source::Levels(levels) => levels
                    .iter()
                    .flat_map(|level| level.iter())
                    .map(|n| n.index() as u32)
                    .collect(),
                Source::Sorted(members) => members.clone(),
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
                Source::Levels(levels) => {
                    for level in levels.iter() {
                        for n in level.iter() {
                            out.extend_from_slice(&(n.index() as u32).to_le_bytes());
                        }
                    }
                }
                Source::Sorted(members) => {
                    for n in members.iter() {
                        out.extend_from_slice(&n.to_le_bytes());
                    }
                }
            }
        }
    }

    let mut entries: Vec<(u64, Source<'_>)> = Vec::new();
    for (name, view) in store.iter() {
        let mut src = match view {
            TypeNodesRef::Overlay(s) => Source::Vec(s),
            TypeNodesRef::Mmap(s) => Source::Slice(s),
            TypeNodesRef::Layered(levels) => Source::Levels(levels),
        };
        let Some(key) = interner.try_resolve_to_key(name) else {
            if src.len() == 0 {
                continue;
            }
            return Err(format!(
                "type index for type '{name}' holds {} nodes but the type name \
                 is not in the graph's interner; refusing to write a \
                 type_indices.bin that cannot be read back",
                src.len()
            ));
        };
        // Ascending order is the file's contract (see the fn doc). The
        // already-ordered case — every save that never deleted, and every
        // untouched mmap base, i.e. the Wikidata-scale one — pays one linear
        // compare pass over bytes the writer is about to copy anyway, and
        // allocates nothing.
        if !src.is_strictly_increasing() {
            let mut members = src.to_members();
            members.sort_unstable();
            if let Some(pair) = members.windows(2).find(|pair| pair[0] == pair[1]) {
                return Err(format!(
                    "type index for type '{name}' lists node {} twice; refusing \
                     to write a type_indices.bin that cannot be read back",
                    pair[0]
                ));
            }
            src = Source::Sorted(members);
        }
        entries.push((key.as_u64(), src));
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
#[path = "type_index_positional_tests.rs"]
mod positional_tests;

#[cfg(test)]
mod validation_tests {
    use super::*;
    use crate::graph::storage::disk::temp_owner::{TempGraphDir, TrackedOwner};

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

    /// A loaded [`TypeIndexBase`] together with the temp directory its
    /// `mmap` points into. Field order is the contract: `base` drops before
    /// `temp`, and `temp`'s guard asserts it.
    ///
    /// The previous helper returned the base alone, so the `TempDir` local
    /// was dropped the moment `load` returned and every assertion below ran
    /// against an unlinked inode — valid on Unix, and therefore silent.
    struct LoadedIndex {
        base: TrackedOwner<TypeIndexBase>,
        /// Held only for its `Drop`: it asserts `base` above is gone.
        _temp: TempGraphDir,
    }

    impl LoadedIndex {
        fn base(&self) -> &TypeIndexBase {
            &self.base
        }
    }

    // `load(...).unwrap_err()` needs `Debug` on the success type; the guard
    // and the mmap-backed base have nothing useful to print.
    impl std::fmt::Debug for LoadedIndex {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("LoadedIndex")
        }
    }

    fn load(bytes: &[u8], interner: &StringInterner) -> std::io::Result<Option<LoadedIndex>> {
        let temp = TempGraphDir::new();
        std::fs::write(temp.path().join("type_indices.bin"), bytes).unwrap();
        let Some(base) = TypeIndexBase::load_from(temp.path(), interner)? else {
            return Ok(None);
        };
        let base = temp.own("TypeIndexBase", base);
        Ok(Some(LoadedIndex { base, _temp: temp }))
    }

    #[test]
    fn valid_little_endian_fixture_round_trips() {
        let mut interner = StringInterner::new();
        let key = interner.get_or_intern("Person").as_u64();
        let loaded = load(&fixture(key, &[1, 7, 42]), &interner)
            .unwrap()
            .unwrap();
        assert_eq!(
            loaded.base().materialize("Person").unwrap(),
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

    /// The writer resolves names against the interner it is handed, so the
    /// directory it emits is always readable by the snapshot's own interner
    /// sidecar. An unregistered name is dropped when empty and fails the save
    /// when populated (a type index is not a rebuildable cache — silently
    /// dropping a populated one would hide every node of that type).
    #[test]
    fn writer_never_emits_a_key_the_interner_cannot_resolve() {
        let temp = tempfile::tempdir().unwrap();
        let mut interner = StringInterner::new();
        interner.get_or_intern("Known");

        let mut store = TypeIndexStore::default();
        store.replace_with(HashMap::from([
            ("Known".to_string(), vec![NodeIndex::new(0)]),
            ("Unregistered".to_string(), Vec::new()),
        ]));
        write_type_indices_bin(temp.path(), &store, &interner).unwrap();

        let base = TypeIndexBase::load_from(temp.path(), &interner)
            .expect("the written directory must load")
            .unwrap();
        assert_eq!(base.materialize("Known").unwrap(), vec![NodeIndex::new(0)]);
        assert!(base.materialize("Unregistered").is_none());

        store.replace_with(HashMap::from([(
            "Unregistered".to_string(),
            vec![NodeIndex::new(0)],
        )]));
        let error = write_type_indices_bin(temp.path(), &store, &interner).unwrap_err();
        assert!(error.contains("Unregistered"), "{error}");
        assert!(error.contains("cannot be read back"), "{error}");
    }

    /// A bucket left out of order by a delete-then-create must still be
    /// written as a payload the reader accepts.
    ///
    /// The bucket is appended to in creation order and a create takes the
    /// slot the delete freed, so `[0, 2, 1]` is the three-node shape of
    /// "create three, delete the middle one, create one more". The reader
    /// requires strict ascension, so before the writer normalized the payload
    /// this save succeeded and the graph could never be loaded again — the
    /// save reported success while destroying the only copy.
    #[test]
    fn a_bucket_left_out_of_order_by_a_reused_slot_still_writes_a_loadable_file() {
        let temp = tempfile::tempdir().unwrap();
        let mut interner = StringInterner::new();
        interner.get_or_intern("Item");

        let mut store = TypeIndexStore::new();
        // create 0,1,2 → delete 1 → create into the freed slot 1.
        for i in [0usize, 1, 2] {
            store.push_to_type("Item", NodeIndex::new(i));
        }
        store.retain_in_type("Item", |idx| idx.index() != 1);
        store.push_to_type("Item", NodeIndex::new(1));
        assert_eq!(
            store.get("Item").unwrap().to_vec(),
            [0, 2, 1].map(NodeIndex::new).to_vec(),
            "precondition: the bucket must actually be out of order, or this \
             test proves nothing"
        );

        write_type_indices_bin(temp.path(), &store, &interner)
            .expect("the save must not refuse a legitimately reordered bucket");
        let base = TypeIndexBase::load_from(temp.path(), &interner)
            .expect("the written file must load — this is the data-loss bug")
            .unwrap();
        assert_eq!(
            base.materialize("Item").unwrap(),
            [0, 1, 2].map(NodeIndex::new).to_vec()
        );
    }

    /// Same shape through the layered (forked) bucket, which reaches the
    /// writer as `TypeNodesRef::Layered` and has its own emit arm.
    #[test]
    fn a_layered_bucket_out_of_order_across_levels_still_writes_a_loadable_file() {
        let temp = tempfile::tempdir().unwrap();
        let mut interner = StringInterner::new();
        interner.get_or_intern("Item");

        let mut store = TypeIndexStore::new();
        for i in [0usize, 2, 3] {
            store.push_to_type("Item", NodeIndex::new(i));
        }
        // A held reader forces the next append into a fresh level, so the
        // out-of-order member lands in a *different* level from its peers.
        let _reader = store.clone();
        store.push_to_type("Item", NodeIndex::new(1));
        assert!(
            matches!(store.get("Item"), Some(TypeNodesRef::Layered(_))),
            "precondition: the bucket must be layered, or this exercises the \
             flat arm again"
        );

        write_type_indices_bin(temp.path(), &store, &interner).unwrap();
        let base = TypeIndexBase::load_from(temp.path(), &interner)
            .expect("the written file must load")
            .unwrap();
        assert_eq!(
            base.materialize("Item").unwrap(),
            [0, 1, 2, 3].map(NodeIndex::new).to_vec()
        );
    }

    /// A bucket holding one node twice is not a reordering — sorting it would
    /// still produce a payload the reader rejects. Fail the *save*, where the
    /// message can name the type, rather than shipping a file that only fails
    /// on the next load.
    #[test]
    fn a_duplicated_member_fails_the_save_instead_of_the_next_load() {
        let temp = tempfile::tempdir().unwrap();
        let mut interner = StringInterner::new();
        interner.get_or_intern("Item");

        let mut store = TypeIndexStore::new();
        store.replace_with(HashMap::from([(
            "Item".to_string(),
            vec![NodeIndex::new(2), NodeIndex::new(0), NodeIndex::new(2)],
        )]));
        let error = write_type_indices_bin(temp.path(), &store, &interner).unwrap_err();
        assert!(error.contains("Item"), "{error}");
        assert!(error.contains("twice"), "{error}");
    }

    /// An already-ascending payload must reach the file byte-for-byte, with no
    /// reordering pass changing it — the normalization is a repair, not a
    /// rewrite of the ordinary save.
    #[test]
    fn an_already_ascending_payload_is_written_unchanged() {
        let temp = tempfile::tempdir().unwrap();
        let mut interner = StringInterner::new();
        interner.get_or_intern("Item");

        let mut store = TypeIndexStore::new();
        for i in [0usize, 4, 9] {
            store.push_to_type("Item", NodeIndex::new(i));
        }
        write_type_indices_bin(temp.path(), &store, &interner).unwrap();
        let base = TypeIndexBase::load_from(temp.path(), &interner)
            .unwrap()
            .unwrap();
        assert_eq!(
            base.materialize("Item").unwrap(),
            [0, 4, 9].map(NodeIndex::new).to_vec()
        );
    }
}

#[cfg(test)]
mod layered_view_tests {
    use super::*;

    /// A store whose bucket is layered must answer every read exactly as the
    /// unlayered store it forked from.
    ///
    /// The `Layered` view walks a stack of slices instead of indexing one, so
    /// every accessor has a second implementation — this pins them against the
    /// original rather than against itself.
    #[test]
    fn a_layered_bucket_answers_exactly_as_the_flat_one() {
        let mut flat = TypeIndexStore::new();
        for i in 0..6u32 {
            flat.push_to_type("Item", NodeIndex::new(i as usize));
        }

        // Fork, then write on both sides so each holds a real level stack.
        let mut forked = flat.clone();
        for i in 6..9u32 {
            forked.push_to_type("Item", NodeIndex::new(i as usize));
            flat.push_to_type("Item", NodeIndex::new(i as usize));
        }
        // A second fork-then-write, so the stack is three levels deep.
        let deeper = forked.clone();
        forked.push_to_type("Item", NodeIndex::new(9));

        let expected: Vec<NodeIndex> = (0..10).map(NodeIndex::new).collect();
        let view = forked.get("Item").expect("bucket present");
        assert!(
            matches!(view, TypeNodesRef::Layered(_)),
            "the fixture must actually be layered, or this test proves nothing"
        );

        assert_eq!(view.len(), expected.len());
        assert!(!view.is_empty());
        assert_eq!(view.to_vec(), expected);
        assert_eq!(view.iter().collect::<Vec<_>>(), expected);
        assert_eq!(view.iter().len(), expected.len());
        for (i, idx) in expected.iter().enumerate() {
            assert_eq!(view.get(i), Some(*idx), "positional read {i}");
            assert!(view.contains(idx));
            assert!(view.binary_search_idx(*idx));
        }
        assert_eq!(view.get(expected.len()), None);
        assert!(!view.contains(&NodeIndex::new(99)));
        assert!(!view.binary_search_idx(NodeIndex::new(99)));

        // `len()` must stay right part-way through a walk, because
        // `ExactSizeIterator` is a promise callers size their buffers on.
        let mut walk = view.iter();
        for remaining in (0..expected.len()).rev() {
            walk.next();
            assert_eq!(walk.len(), remaining);
        }

        // The other holders are unaffected by either write.
        assert_eq!(
            deeper.get("Item").unwrap().to_vec(),
            expected[..9].to_vec(),
            "the intermediate fork sees its own snapshot"
        );
        assert_eq!(flat.get("Item").unwrap().to_vec(), expected[..9].to_vec());
    }

    /// A layered bucket must survive the save writer unchanged — the on-disk
    /// payload is the merged content, in order.
    #[test]
    fn the_writer_flattens_a_layered_bucket_in_order() {
        let temp = tempfile::tempdir().unwrap();
        let mut interner = StringInterner::new();
        interner.get_or_intern("Item");

        let mut store = TypeIndexStore::new();
        for i in 0..4u32 {
            store.push_to_type("Item", NodeIndex::new(i as usize));
        }
        let _reader = store.clone();
        store.push_to_type("Item", NodeIndex::new(4));
        assert!(matches!(store.get("Item"), Some(TypeNodesRef::Layered(_))));

        write_type_indices_bin(temp.path(), &store, &interner).unwrap();
        let base = TypeIndexBase::load_from(temp.path(), &interner)
            .expect("written directory must load")
            .unwrap();
        assert_eq!(
            base.materialize("Item").unwrap(),
            (0..5).map(NodeIndex::new).collect::<Vec<_>>()
        );
    }
}
