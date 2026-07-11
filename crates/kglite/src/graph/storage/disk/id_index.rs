//! Mmap-resident `id_indices.bin` store with overlay for mutations.
//!
//! Replaces the eager `zstd::decode_all` + 124M-entry `HashMap::insert`
//! load path. Reads come from a memory-mapped flat binary on the disk;
//! mutations land in an in-memory overlay that takes precedence over
//! the base. On save, overlay + base are merged into a fresh `.bin`.
//!
//! ## File format `id_indices.bin`
//!
//! ```text
//! Header (32 bytes):
//!   [ 0.. 8]  magic           = b"KGLIIDXR"  (R = raw, mmap-friendly)
//!   [ 8..12]  version         = u32 LE (= 1)
//!   [12..16]  num_types       = u32 LE
//!   [16..24]  dir_offset      = u64 LE   (always 32)
//!   [24..32]  data_offset     = u64 LE   (32 + 48 * num_types)
//!
//! Directory at [dir_offset]: 48 bytes per entry, sorted by type_key:
//!   [ 0.. 8]  type_key:    u64 LE  (InternedKey)
//!   [ 8.. 9]  variant:     u8      (0 = Integer, 1 = General)
//!   [ 9..16]  padding:     [u8; 7]
//!   [16..24]  num_entries: u64 LE
//!   [24..32]  payload_off: u64 LE   (file-relative)
//!   [32..40]  payload_len: u64 LE
//!   [40..48]  padding:     u64
//!
//! Data section at [data_offset]:
//!   Integer (variant=0):
//!     [payload_off..payload_off + 4*num_entries]               keys: [u32 sorted asc]
//!     [payload_off + 4*num_entries..payload_off + payload_len] idxs: [u32]
//!   General (variant=1):
//!     bincode of HashMap<Value, NodeIndex>, length = payload_len
//! ```
//!
//! Lookup is `O(log n)` binary search on `keys` for the Integer variant
//! (cache-friendly, ~24 comparisons even at 13M entries) and a single
//! `HashMap` probe for the General variant (lazily deserialized).

use crate::datatypes::Value;
use crate::graph::schema::{InternedKey, StringInterner, TypeIdIndex};
use bincode::Options;
use memmap2::Mmap;
use petgraph::graph::NodeIndex;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

const MAGIC: &[u8; 8] = b"KGLIIDXR";
const VERSION: u32 = 1;
const HEADER_BYTES: usize = 32;
const DIR_ENTRY_BYTES: usize = 48;
const MAX_GENERAL_INDEX_DECODE_BYTES: u64 = 2 * 1024 * 1024 * 1024;

fn invalid_index(message: &str) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("invalid id_indices.bin: {message}"),
    )
}

fn read_le_u32(bytes: &[u8], index: usize) -> Option<u32> {
    let start = index.checked_mul(4)?;
    Some(u32::from_le_bytes(
        bytes.get(start..start.checked_add(4)?)?.try_into().ok()?,
    ))
}

fn le_u32_binary_search(bytes: &[u8], wanted: u32) -> Option<usize> {
    let mut low = 0usize;
    let mut high = bytes.len() / 4;
    while low < high {
        let mid = low + (high - low) / 2;
        match read_le_u32(bytes, mid)?.cmp(&wanted) {
            std::cmp::Ordering::Less => low = mid + 1,
            std::cmp::Ordering::Greater => high = mid,
            std::cmp::Ordering::Equal => return Some(mid),
        }
    }
    None
}

/// Mmap-backed read-only view of `id_indices.bin`.
pub struct IdIndexBase {
    mmap: Arc<Mmap>,
    /// type_name -> directory entry. Built once at load (88k entries × ~50 bytes ≈ 4 MB).
    /// Strings owned to keep the API HashMap-compatible without lifetime gymnastics.
    dir: HashMap<String, BaseEntry>,
    /// Lazy materialization cache for General variant (deserialized bincode blobs).
    /// Integer variant never enters here — it's read directly from mmap.
    general_cache: RwLock<HashMap<String, Arc<HashMap<Value, NodeIndex>>>>,
}

#[derive(Clone, Copy)]
struct BaseEntry {
    variant: u8,
    num_entries: u32,
    payload_off: u64,
    payload_len: u64,
}

impl IdIndexBase {
    /// Load `id_indices.bin` from `dir`. Returns `Ok(None)` if absent or magic mismatch.
    pub fn load_from(dir: &Path, interner: &StringInterner) -> std::io::Result<Option<Self>> {
        let path = dir.join("id_indices.bin");
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
        let dir_offset = usize::try_from(u64::from_le_bytes(mmap[16..24].try_into().unwrap()))
            .map_err(|_| invalid_index("directory offset exceeds usize"))?;
        let data_offset = usize::try_from(u64::from_le_bytes(mmap[24..32].try_into().unwrap()))
            .map_err(|_| invalid_index("data offset exceeds usize"))?;
        let dir_bytes = DIR_ENTRY_BYTES
            .checked_mul(num_types)
            .ok_or_else(|| invalid_index("directory size overflow"))?;
        let need = dir_offset
            .checked_add(dir_bytes)
            .ok_or_else(|| invalid_index("directory range overflow"))?;
        if dir_offset != HEADER_BYTES || data_offset != need || need > len {
            return Err(invalid_index("invalid directory/data boundary"));
        }

        let mut dir_map: HashMap<String, BaseEntry> = HashMap::with_capacity(num_types);
        let mut general_cache_map = HashMap::new();
        let mut previous_key = None;
        let mut expected_payload = data_offset;
        for i in 0..num_types {
            let off = dir_offset + i * DIR_ENTRY_BYTES;
            let type_key = u64::from_le_bytes(mmap[off..off + 8].try_into().unwrap());
            let variant = mmap[off + 8];
            let num_entries_u64 = u64::from_le_bytes(mmap[off + 16..off + 24].try_into().unwrap());
            let payload_off = u64::from_le_bytes(mmap[off + 24..off + 32].try_into().unwrap());
            let payload_len = u64::from_le_bytes(mmap[off + 32..off + 40].try_into().unwrap());
            if previous_key.is_some_and(|previous| type_key <= previous) {
                return Err(invalid_index("directory keys are not strictly increasing"));
            }
            previous_key = Some(type_key);
            if !matches!(variant, 0 | 1) {
                return Err(invalid_index("directory contains an unknown variant"));
            }
            let num_entries = u32::try_from(num_entries_u64)
                .map_err(|_| invalid_index("entry count exceeds u32"))?;
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
            if variant == 0 {
                let expected_len = num_entries_u64
                    .checked_mul(8)
                    .ok_or_else(|| invalid_index("integer payload size overflow"))?;
                if payload_len != expected_len {
                    return Err(invalid_index("integer payload has invalid size"));
                }
                let keys_end = payload_off_usize + num_entries as usize * 4;
                let mut previous = None;
                for index in 0..num_entries as usize {
                    let key = read_le_u32(&mmap[payload_off_usize..keys_end], index).unwrap();
                    if previous.is_some_and(|prior| key <= prior) {
                        return Err(invalid_index("integer keys are not strictly increasing"));
                    }
                    previous = Some(key);
                }
            } else if payload_len > MAX_GENERAL_INDEX_DECODE_BYTES {
                return Err(invalid_index("general payload exceeds decode limit"));
            }
            expected_payload = payload_end;
            let name = interner
                .try_resolve(InternedKey::from_u64(type_key))
                .ok_or_else(|| invalid_index("directory contains an unresolved type key"))?;
            if variant == 1 {
                let blob = &mmap[payload_off_usize..payload_end];
                let encoded_count = blob
                    .get(..8)
                    .map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()))
                    .ok_or_else(|| invalid_index("general payload is truncated before count"))?;
                if encoded_count != num_entries_u64 {
                    return Err(invalid_index(
                        "general payload count disagrees with directory",
                    ));
                }
                let minimum = 8u64
                    .checked_add(
                        num_entries_u64
                            .checked_mul(8)
                            .ok_or_else(|| invalid_index("general minimum size overflow"))?,
                    )
                    .ok_or_else(|| invalid_index("general minimum size overflow"))?;
                if payload_len < minimum {
                    return Err(invalid_index(
                        "general payload cannot contain declared entries",
                    ));
                }
                let map: HashMap<Value, NodeIndex> = bincode::options()
                    .with_fixint_encoding()
                    .with_little_endian()
                    .reject_trailing_bytes()
                    .with_limit(MAX_GENERAL_INDEX_DECODE_BYTES)
                    .deserialize(blob)
                    .map_err(|_| invalid_index("general payload bincode is malformed"))?;
                if map.len() != num_entries as usize {
                    return Err(invalid_index(
                        "general payload has duplicate or missing keys",
                    ));
                }
                general_cache_map.insert(name.to_string(), Arc::new(map));
            }
            if dir_map
                .insert(
                    name.to_string(),
                    BaseEntry {
                        variant,
                        num_entries,
                        payload_off,
                        payload_len,
                    },
                )
                .is_some()
            {
                return Err(invalid_index("duplicate resolved type name"));
            }
        }
        if expected_payload != len {
            return Err(invalid_index(
                "payload directory does not cover the file exactly",
            ));
        }

        Ok(Some(Self {
            mmap: Arc::new(mmap),
            dir: dir_map,
            general_cache: RwLock::new(general_cache_map),
        }))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.dir.contains_key(name)
    }

    pub fn lookup(&self, name: &str, id: &Value) -> Option<NodeIndex> {
        let entry = self.dir.get(name)?;
        match entry.variant {
            0 => self.lookup_integer(entry, id),
            1 => self.lookup_general(name, entry, id),
            _ => None,
        }
    }

    /// Materialize a base entry into an owned `TypeIdIndex` (used on save and
    /// on first mutation when the entry must be promoted into the overlay).
    pub fn materialize(&self, name: &str) -> Option<TypeIdIndex> {
        let entry = self.dir.get(name)?;
        match entry.variant {
            0 => {
                let (keys, idxs) = self.integer_bytes(entry)?;
                let mut map: HashMap<u32, NodeIndex> =
                    HashMap::with_capacity(entry.num_entries as usize);
                for index in 0..entry.num_entries as usize {
                    map.insert(
                        read_le_u32(keys, index)?,
                        NodeIndex::new(read_le_u32(idxs, index)? as usize),
                    );
                }
                Some(TypeIdIndex::Integer(map))
            }
            1 => {
                let map = self.general_map(name, entry)?;
                Some(TypeIdIndex::General((*map).clone()))
            }
            _ => None,
        }
    }

    fn integer_bytes(&self, entry: &BaseEntry) -> Option<(&[u8], &[u8])> {
        let n = entry.num_entries as usize;
        let off = entry.payload_off as usize;
        let half = n * 4;
        if entry.payload_len != (half * 2) as u64 {
            return None;
        }
        let bytes = self.mmap.get(off..off + half * 2)?;
        Some(bytes.split_at(half))
    }

    fn lookup_integer(&self, entry: &BaseEntry, id: &Value) -> Option<NodeIndex> {
        let key_u32 = coerce_to_u32(id)?;
        let (keys, idxs) = self.integer_bytes(entry)?;
        let index = le_u32_binary_search(keys, key_u32)?;
        Some(NodeIndex::new(read_le_u32(idxs, index)? as usize))
    }

    fn lookup_general(&self, name: &str, entry: &BaseEntry, id: &Value) -> Option<NodeIndex> {
        let map = self.general_map(name, entry)?;
        if let Some(&idx) = map.get(id) {
            return Some(idx);
        }
        // Mirror TypeIdIndex::General numeric coercion fallbacks (Int64 ↔
        // UniqueId, Float64 → Int/UniqueId). No string→u32 coercion — a
        // String id matches only by exact value (handled above).
        match id {
            Value::Int64(i) => {
                if *i >= 0 && *i <= u32::MAX as i64 {
                    return map.get(&Value::UniqueId(*i as u32)).copied();
                }
                None
            }
            Value::UniqueId(u) => map.get(&Value::Int64(*u as i64)).copied(),
            Value::Float64(f) => {
                if f.fract() == 0.0 {
                    let i = *f as i64;
                    if let Some(&idx) = map.get(&Value::Int64(i)) {
                        return Some(idx);
                    }
                    if i >= 0 && i <= u32::MAX as i64 {
                        return map.get(&Value::UniqueId(i as u32)).copied();
                    }
                }
                None
            }
            _ => None,
        }
    }

    fn general_map(&self, name: &str, entry: &BaseEntry) -> Option<Arc<HashMap<Value, NodeIndex>>> {
        if let Some(arc) = self.general_cache.read().unwrap().get(name).cloned() {
            return Some(arc);
        }
        let off = entry.payload_off as usize;
        let len = entry.payload_len as usize;
        let blob = self.mmap.get(off..off + len)?;
        let map: HashMap<Value, NodeIndex> = bincode::options()
            .with_fixint_encoding()
            .with_little_endian()
            .reject_trailing_bytes()
            .with_limit(MAX_GENERAL_INDEX_DECODE_BYTES)
            .deserialize(blob)
            .ok()?;
        if map.len() != entry.num_entries as usize {
            return None;
        }
        let arc = Arc::new(map);
        self.general_cache
            .write()
            .unwrap()
            .insert(name.to_string(), Arc::clone(&arc));
        Some(arc)
    }
}

/// HashMap-shaped wrapper around an optional mmap base + in-memory overlay.
///
/// Reads consult overlay first (covers post-load mutations), then base.
/// Mutations only ever land in overlay; `removed` tracks types that the
/// caller explicitly cleared so that base entries are masked.
#[derive(Default)]
pub struct IdIndexStore {
    /// In-memory layer: indices built/mutated post-load, plus lazily-cached
    /// indices the read path builds on a miss. Behind a `RwLock` so the
    /// read path can build + cache through `&self` — `DirGraph` is shared
    /// as `Arc<DirGraph>` and reads run on multiple threads (GIL-release),
    /// so this must be thread-safe.
    overlay: RwLock<HashMap<String, TypeIdIndex>>,
    /// Types that exist in `base` but were removed/invalidated post-load.
    removed: std::collections::HashSet<String>,
    base: Option<Arc<IdIndexBase>>,
}

impl Clone for IdIndexStore {
    fn clone(&self) -> Self {
        Self {
            overlay: RwLock::new(self.overlay.read().unwrap().clone()),
            removed: self.removed.clone(),
            base: self.base.clone(),
        }
    }
}

impl IdIndexStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_base(base: IdIndexBase) -> Self {
        Self {
            overlay: RwLock::new(HashMap::new()),
            removed: std::collections::HashSet::new(),
            base: Some(Arc::new(base)),
        }
    }

    pub fn contains_key(&self, name: &str) -> bool {
        if self.overlay.read().unwrap().contains_key(name) {
            return true;
        }
        if self.removed.contains(name) {
            return false;
        }
        self.base.as_ref().is_some_and(|b| b.contains(name))
    }

    /// Look up `id` for `name`. If the type isn't indexed anywhere (neither
    /// overlay nor base, or it was invalidated), build the index via `build`
    /// — which scans the graph — and cache it in the overlay, so the read
    /// path is O(1) on every subsequent lookup. The build runs at most once
    /// per type until the next invalidation. Returns None when the id simply
    /// isn't present (no scan).
    ///
    /// This is the fix for the O(node-position) linear-scan footgun: the
    /// read path (`MATCH (n {id:X})`, `MERGE` match) used to fall back to a
    /// full scan whenever the index was absent (after `add_nodes` /
    /// `CREATE` / `DELETE`). Now it self-heals on first read, regardless of
    /// how the graph was built or mutated. (issue #20)
    pub fn lookup_or_build(
        &self,
        name: &str,
        id: &Value,
        build: impl FnOnce() -> TypeIdIndex,
    ) -> Option<NodeIndex> {
        // Fast path: already cached in the overlay.
        {
            let ov = self.overlay.read().unwrap();
            if let Some(idx) = ov.get(name) {
                return idx.get(id);
            }
        }
        // Base (mmap) layer, unless explicitly invalidated.
        if !self.removed.contains(name) {
            if let Some(base) = self.base.as_deref() {
                if base.contains(name) {
                    return base.lookup(name, id);
                }
            }
        }
        // Not indexed anywhere — build once and cache (idempotent under a
        // concurrent race: the first writer wins, both indices are equal).
        let built = build();
        let mut ov = self.overlay.write().unwrap();
        ov.entry(name.to_string()).or_insert(built).get(id)
    }

    /// Look up without building — None when the type isn't indexed.
    pub fn lookup(&self, name: &str, id: &Value) -> Option<NodeIndex> {
        {
            let ov = self.overlay.read().unwrap();
            if let Some(idx) = ov.get(name) {
                return idx.get(id);
            }
        }
        if self.removed.contains(name) {
            return None;
        }
        self.base.as_deref().and_then(|b| {
            if b.contains(name) {
                b.lookup(name, id)
            } else {
                None
            }
        })
    }

    /// Materialize the full `id → NodeIndex` map for a type, or None when the
    /// type isn't indexed. Used by the add_nodes conflict-check fast path.
    pub fn materialize_type(&self, name: &str) -> Option<HashMap<Value, NodeIndex>> {
        {
            let ov = self.overlay.read().unwrap();
            if let Some(idx) = ov.get(name) {
                return Some(idx.iter().collect());
            }
        }
        if self.removed.contains(name) {
            return None;
        }
        let base = self.base.as_deref()?;
        if base.contains(name) {
            base.materialize(name).map(|ti| ti.iter().collect())
        } else {
            None
        }
    }

    pub fn insert(&mut self, name: String, idx: TypeIdIndex) {
        self.removed.remove(&name);
        self.overlay.get_mut().unwrap().insert(name, idx);
    }

    pub fn remove(&mut self, name: &str) -> Option<TypeIdIndex> {
        let prev = self.overlay.get_mut().unwrap().remove(name);
        if self.base.as_ref().is_some_and(|b| b.contains(name)) {
            self.removed.insert(name.to_string());
        }
        prev
    }

    pub fn clear(&mut self) {
        self.overlay.get_mut().unwrap().clear();
        if let Some(base) = &self.base {
            self.removed.extend(base.dir.keys().cloned());
        }
    }

    pub fn len(&self) -> usize {
        let overlay = self.overlay.read().unwrap();
        let base_count = self
            .base
            .as_ref()
            .map(|b| b.dir.keys().filter(|k| !self.removed.contains(*k)).count())
            .unwrap_or(0);
        let overlay_only = overlay
            .keys()
            .filter(|k| self.base.as_ref().map(|b| !b.contains(k)).unwrap_or(true))
            .count();
        base_count + overlay_only
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Owned snapshot of every live `TypeIdIndex` (overlay first, then base
    /// entries that aren't shadowed/removed). Cold path — used by save and
    /// N-Triples export. Returns owned indices because the read lock can't
    /// be held across the caller's iteration.
    pub fn values(&self) -> Vec<TypeIdIndex> {
        self.snapshot().into_iter().map(|(_, v)| v).collect()
    }

    /// Owned `(name, TypeIdIndex)` snapshot of every live entry. Cold path.
    pub fn iter(&self) -> Vec<(String, TypeIdIndex)> {
        self.snapshot()
    }

    fn snapshot(&self) -> Vec<(String, TypeIdIndex)> {
        let overlay = self.overlay.read().unwrap();
        // Overlay entries first, then base entries that aren't shadowed.
        let mut out: Vec<(String, TypeIdIndex)> = overlay
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        if let Some(base) = self.base.as_deref() {
            for k in base.dir.keys() {
                if !overlay.contains_key(k.as_str()) && !self.removed.contains(k.as_str()) {
                    if let Some(materialized) = base.materialize(k) {
                        out.push((k.clone(), materialized));
                    }
                }
            }
        }
        out
    }

    /// HashMap-`entry`-shaped accessor: materialize any base entry into the
    /// overlay (or default-construct), then hand back a `&mut` to it. Used by
    /// the N-Triples loader's per-entity incremental build. `&mut self` gives
    /// exclusive access, so `get_mut()` is uncontended (no lock cost).
    pub fn entry_or_default(&mut self, name: String) -> &mut TypeIdIndex {
        let needs_materialize = {
            let overlay = self.overlay.get_mut().unwrap();
            !overlay.contains_key(&name) && !self.removed.contains(&name)
        };
        if needs_materialize {
            if let Some(base) = self.base.as_deref() {
                if let Some(materialized) = base.materialize(&name) {
                    self.overlay
                        .get_mut()
                        .unwrap()
                        .insert(name.clone(), materialized);
                }
            }
        }
        self.removed.remove(&name);
        self.overlay.get_mut().unwrap().entry(name).or_default()
    }

    /// Replace the entire store with a fresh HashMap (used by load fallback
    /// for legacy `.bin.zst`-only graphs and by `reindex()`).
    pub fn replace_with(&mut self, map: HashMap<String, TypeIdIndex>) {
        *self.overlay.get_mut().unwrap() = map;
        self.removed.clear();
        self.base = None;
    }
}

/// Coerce a `Value` to `u32` for binary search on the Integer variant.
/// Mirrors the matching branches in `TypeIdIndex::get`.
fn coerce_to_u32(id: &Value) -> Option<u32> {
    match id {
        Value::UniqueId(u) => Some(*u),
        Value::Int64(i) => {
            if *i >= 0 && *i <= u32::MAX as i64 {
                Some(*i as u32)
            } else {
                None
            }
        }
        Value::Float64(f) => {
            if f.fract() == 0.0 {
                let i = *f as i64;
                if i >= 0 && i <= u32::MAX as i64 {
                    Some(i as u32)
                } else {
                    None
                }
            } else {
                None
            }
        }
        // No string→u32 coercion: a String id matches only by exact value.
        _ => None,
    }
}

// =============================================================================
// Writer
// =============================================================================

/// Write `id_indices.bin` (raw mmap layout). Iterates the store's union view
/// (overlay + base) so saves capture both fresh mutations and unchanged
/// base entries.
pub fn write_id_indices_bin(
    dir: &Path,
    store: &IdIndexStore,
    interner: &StringInterner,
) -> Result<(), String> {
    // Collect (type_key, name, materialized) triples sorted by type_key.
    // `store.iter()` already returns owned, fully-materialized indices
    // (overlay + unshadowed base).
    let mut entries: Vec<(u64, String, TypeIdIndex)> = Vec::new();
    let mut interner_clone = interner.clone();
    for (name, materialized) in store.iter() {
        let key = interner_clone
            .try_get_or_intern(&name)
            .map_err(|e| e.to_string())?
            .as_u64();
        entries.push((key, name, materialized));
    }
    entries.sort_by_key(|(k, _, _)| *k);

    let num_types = entries.len();
    let header_size = HEADER_BYTES;
    let dir_size = DIR_ENTRY_BYTES * num_types;
    let data_offset = header_size + dir_size;

    // Pre-compute payload offsets/lengths so we can emit the directory first.
    struct Plan {
        type_key: u64,
        variant: u8,
        num_entries: u64,
        payload_off: u64,
        payload_len: u64,
        data: Vec<u8>,
    }

    let mut plans: Vec<Plan> = Vec::with_capacity(num_types);
    let mut cursor = data_offset as u64;

    for (type_key, _name, idx) in &entries {
        match idx {
            TypeIdIndex::Integer(map) => {
                let mut pairs: Vec<(u32, u32)> =
                    map.iter().map(|(k, v)| (*k, v.index() as u32)).collect();
                pairs.sort_by_key(|(k, _)| *k);
                let n = pairs.len();
                let mut data = Vec::with_capacity(n * 8);
                for (k, _) in &pairs {
                    data.extend_from_slice(&k.to_le_bytes());
                }
                for (_, v) in &pairs {
                    data.extend_from_slice(&v.to_le_bytes());
                }
                let len = data.len() as u64;
                plans.push(Plan {
                    type_key: *type_key,
                    variant: 0,
                    num_entries: n as u64,
                    payload_off: cursor,
                    payload_len: len,
                    data,
                });
                cursor += len;
            }
            TypeIdIndex::General(map) => {
                let blob = bincode::serialize(map)
                    .map_err(|e| format!("id_indices General-variant bincode failed: {}", e))?;
                let len = blob.len() as u64;
                plans.push(Plan {
                    type_key: *type_key,
                    variant: 1,
                    num_entries: map.len() as u64,
                    payload_off: cursor,
                    payload_len: len,
                    data: blob,
                });
                cursor += len;
            }
        }
    }

    let total = cursor as usize;
    let mut out = Vec::with_capacity(total);
    // Header
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(num_types as u32).to_le_bytes());
    out.extend_from_slice(&(HEADER_BYTES as u64).to_le_bytes());
    out.extend_from_slice(&(data_offset as u64).to_le_bytes());

    // Directory
    for plan in &plans {
        out.extend_from_slice(&plan.type_key.to_le_bytes());
        out.push(plan.variant);
        out.extend_from_slice(&[0u8; 7]);
        out.extend_from_slice(&plan.num_entries.to_le_bytes());
        out.extend_from_slice(&plan.payload_off.to_le_bytes());
        out.extend_from_slice(&plan.payload_len.to_le_bytes());
        out.extend_from_slice(&[0u8; 8]);
    }

    // Data
    for plan in plans {
        out.extend_from_slice(&plan.data);
    }

    debug_assert_eq!(out.len(), total);

    std::fs::write(dir.join("id_indices.bin"), out)
        .map_err(|e| format!("Failed to write id_indices.bin: {}", e))?;
    Ok(())
}

#[cfg(test)]
mod validation_tests {
    use super::*;

    fn integer_fixture(type_key: u64, pairs: &[(u32, u32)]) -> Vec<u8> {
        let data_offset = HEADER_BYTES + DIR_ENTRY_BYTES;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&(HEADER_BYTES as u64).to_le_bytes());
        bytes.extend_from_slice(&(data_offset as u64).to_le_bytes());
        bytes.extend_from_slice(&type_key.to_le_bytes());
        bytes.push(0);
        bytes.extend_from_slice(&[0; 7]);
        bytes.extend_from_slice(&(pairs.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(data_offset as u64).to_le_bytes());
        bytes.extend_from_slice(&((pairs.len() * 8) as u64).to_le_bytes());
        bytes.extend_from_slice(&[0; 8]);
        for (key, _) in pairs {
            bytes.extend_from_slice(&key.to_le_bytes());
        }
        for (_, node) in pairs {
            bytes.extend_from_slice(&node.to_le_bytes());
        }
        bytes
    }

    fn load(bytes: &[u8], interner: &StringInterner) -> std::io::Result<Option<IdIndexBase>> {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("id_indices.bin"), bytes).unwrap();
        IdIndexBase::load_from(temp.path(), interner)
    }

    fn assert_invalid(bytes: &[u8], interner: &StringInterner) {
        let outcome = std::panic::catch_unwind(|| load(bytes, interner));
        match outcome.expect("invalid index must not panic") {
            Err(error) => assert_eq!(error.kind(), std::io::ErrorKind::InvalidData),
            Ok(_) => panic!("invalid index loaded successfully"),
        }
    }

    #[test]
    fn integer_fixture_reads_canonical_little_endian_bytes() {
        let mut interner = StringInterner::new();
        let key = interner.get_or_intern("Person").as_u64();
        let base = load(&integer_fixture(key, &[(7, 70), (42, 420)]), &interner)
            .unwrap()
            .unwrap();
        assert_eq!(
            base.lookup("Person", &Value::UniqueId(7)),
            Some(NodeIndex::new(70))
        );
        assert_eq!(
            base.lookup("Person", &Value::UniqueId(42)),
            Some(NodeIndex::new(420))
        );
    }

    #[test]
    fn rejects_invalid_header_directory_and_variant() {
        let mut interner = StringInterner::new();
        let key = interner.get_or_intern("Person").as_u64();
        let valid = integer_fixture(key, &[(7, 70)]);

        let mut huge_count = valid.clone();
        huge_count[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_invalid(&huge_count, &interner);
        let mut bad_dir = valid.clone();
        bad_dir[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
        assert_invalid(&bad_dir, &interner);
        let mut bad_data = valid.clone();
        bad_data[24..32].copy_from_slice(&33u64.to_le_bytes());
        assert_invalid(&bad_data, &interner);
        let mut bad_variant = valid.clone();
        bad_variant[40] = 2;
        assert_invalid(&bad_variant, &interner);
    }

    #[test]
    fn rejects_bad_counts_ranges_and_integer_ordering() {
        let mut interner = StringInterner::new();
        let key = interner.get_or_intern("Person").as_u64();
        let valid = integer_fixture(key, &[(7, 70), (42, 420)]);

        let mut too_many = valid.clone();
        too_many[48..56].copy_from_slice(&(u32::MAX as u64 + 1).to_le_bytes());
        assert_invalid(&too_many, &interner);
        let mut past_eof = valid.clone();
        past_eof[56..64].copy_from_slice(&u64::MAX.to_le_bytes());
        assert_invalid(&past_eof, &interner);
        let mut wrong_len = valid.clone();
        wrong_len[64..72].copy_from_slice(&15u64.to_le_bytes());
        assert_invalid(&wrong_len, &interner);
        assert_invalid(&integer_fixture(key, &[(42, 1), (7, 2)]), &interner);
        assert_invalid(&integer_fixture(key, &[(7, 1), (7, 2)]), &interner);
    }

    #[test]
    fn rejects_malformed_general_bincode_during_load() {
        let mut interner = StringInterner::new();
        let key = interner.get_or_intern("StringIds").as_u64();
        let mut bytes = integer_fixture(key, &[(1, 1)]);
        bytes[40] = 1;
        bytes[64..72].copy_from_slice(&16u64.to_le_bytes());
        bytes.truncate(HEADER_BYTES + DIR_ENTRY_BYTES);
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&[0xff; 8]);
        assert_invalid(&bytes, &interner);
    }

    #[test]
    fn writer_round_trip_accepts_unaligned_integer_payload_after_general() {
        let temp = tempfile::tempdir().unwrap();
        let mut interner = StringInterner::new();
        let candidates = ["Alpha", "Beta"];
        for name in candidates {
            interner.get_or_intern(name);
        }
        let mut ordered = candidates;
        ordered.sort_by_key(|name| InternedKey::from_str(name).as_u64());
        let general_name = ordered[0];
        let integer_name = ordered[1];
        let general = TypeIdIndex::General(HashMap::from([(
            Value::String("x".into()),
            NodeIndex::new(3),
        )]));
        let integer = TypeIdIndex::Integer(HashMap::from([(7, NodeIndex::new(4))]));
        let mut store = IdIndexStore::default();
        store.replace_with(HashMap::from([
            (general_name.to_string(), general),
            (integer_name.to_string(), integer),
        ]));
        write_id_indices_bin(temp.path(), &store, &interner).unwrap();

        let raw = std::fs::read(temp.path().join("id_indices.bin")).unwrap();
        let second_payload_off = u64::from_le_bytes(
            raw[HEADER_BYTES + DIR_ENTRY_BYTES + 24..HEADER_BYTES + DIR_ENTRY_BYTES + 32]
                .try_into()
                .unwrap(),
        );
        assert_ne!(
            second_payload_off % 4,
            0,
            "fixture must exercise an unaligned v1 integer payload"
        );

        let base = IdIndexBase::load_from(temp.path(), &interner)
            .unwrap()
            .unwrap();
        assert_eq!(
            base.lookup(general_name, &Value::String("x".into())),
            Some(NodeIndex::new(3))
        );
        assert_eq!(
            base.lookup(integer_name, &Value::UniqueId(7)),
            Some(NodeIndex::new(4))
        );
    }

    #[test]
    fn rejects_unsupported_unresolved_and_trailing_data() {
        let mut interner = StringInterner::new();
        let key = interner.get_or_intern("Person").as_u64();
        let valid = integer_fixture(key, &[(7, 70)]);
        let mut version = valid.clone();
        version[8..12].copy_from_slice(&2u32.to_le_bytes());
        assert_invalid(&version, &interner);
        let mut trailing = valid.clone();
        trailing.push(0);
        assert_invalid(&trailing, &interner);
        assert_invalid(&integer_fixture(key.wrapping_add(1), &[(7, 70)]), &interner);
    }
}
