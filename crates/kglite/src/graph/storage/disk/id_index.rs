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
use memmap2::Mmap;
use petgraph::graph::NodeIndex;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

const MAGIC: &[u8; 8] = b"KGLIIDXR";
const VERSION: u32 = 1;
const HEADER_BYTES: usize = 32;
const DIR_ENTRY_BYTES: usize = 48;

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
            return Ok(None);
        }
        let num_types = u32::from_le_bytes(mmap[12..16].try_into().unwrap()) as usize;
        let dir_offset = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
        let _data_offset = u64::from_le_bytes(mmap[24..32].try_into().unwrap()) as usize;

        let need = dir_offset + DIR_ENTRY_BYTES * num_types;
        if len < need {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "id_indices.bin truncated at directory",
            ));
        }

        let mut dir_map: HashMap<String, BaseEntry> = HashMap::with_capacity(num_types);
        for i in 0..num_types {
            let off = dir_offset + i * DIR_ENTRY_BYTES;
            let type_key = u64::from_le_bytes(mmap[off..off + 8].try_into().unwrap());
            let variant = mmap[off + 8];
            let num_entries =
                u64::from_le_bytes(mmap[off + 16..off + 24].try_into().unwrap()) as u32;
            let payload_off = u64::from_le_bytes(mmap[off + 24..off + 32].try_into().unwrap());
            let payload_len = u64::from_le_bytes(mmap[off + 32..off + 40].try_into().unwrap());
            if let Some(name) = interner.try_resolve(InternedKey::from_u64(type_key)) {
                dir_map.insert(
                    name.to_string(),
                    BaseEntry {
                        variant,
                        num_entries,
                        payload_off,
                        payload_len,
                    },
                );
            }
        }

        Ok(Some(Self {
            mmap: Arc::new(mmap),
            dir: dir_map,
            general_cache: RwLock::new(HashMap::new()),
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
                let (keys, idxs) = self.integer_slices(entry)?;
                let mut map: HashMap<u32, NodeIndex> = HashMap::with_capacity(keys.len());
                for (k, v) in keys.iter().zip(idxs.iter()) {
                    map.insert(*k, NodeIndex::new(*v as usize));
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

    fn integer_slices(&self, entry: &BaseEntry) -> Option<(&[u32], &[u32])> {
        let n = entry.num_entries as usize;
        let off = entry.payload_off as usize;
        let half = n * 4;
        if entry.payload_len < (half * 2) as u64 {
            return None;
        }
        let bytes = self.mmap.get(off..off + half * 2)?;
        // SAFETY: payload was written as little-endian u32 arrays at file offsets
        // chosen so each region begins at a 4-byte boundary; mmap pages are
        // page-aligned (4 KB) so any sub-slice that's 4-byte aligned by file
        // construction is also 4-byte aligned in memory.
        // We index two adjacent u32 slices of length `n`, totalling
        // `2 * n * 4` bytes, fully inside `payload_len` (verified above).
        // No mutation possible — `Mmap` (not `MmapMut`) is a read-only mapping.
        // SAFETY: see comment above (4-byte aligned + length-checked).
        unsafe {
            let keys_ptr = bytes.as_ptr() as *const u32;
            let idxs_ptr = bytes.as_ptr().add(half) as *const u32;
            Some((
                std::slice::from_raw_parts(keys_ptr, n),
                std::slice::from_raw_parts(idxs_ptr, n),
            ))
        }
    }

    fn lookup_integer(&self, entry: &BaseEntry, id: &Value) -> Option<NodeIndex> {
        let key_u32 = coerce_to_u32(id)?;
        let (keys, idxs) = self.integer_slices(entry)?;
        keys.binary_search(&key_u32)
            .ok()
            .map(|i| NodeIndex::new(idxs[i] as usize))
    }

    fn lookup_general(&self, name: &str, entry: &BaseEntry, id: &Value) -> Option<NodeIndex> {
        let map = self.general_map(name, entry)?;
        if let Some(&idx) = map.get(id) {
            return Some(idx);
        }
        // Mirror TypeIdIndex::General coercion fallbacks (Int64 ↔ UniqueId,
        // Float64 → Int/UniqueId, prefix-stripped String → UniqueId).
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
            Value::String(s) => strip_prefix_to_u32(s).and_then(|u| {
                map.get(&Value::UniqueId(u))
                    .or_else(|| map.get(&Value::Int64(u as i64)))
                    .copied()
            }),
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
        let map: HashMap<Value, NodeIndex> = bincode::deserialize(blob).ok()?;
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
#[derive(Default, Clone)]
pub struct IdIndexStore {
    overlay: HashMap<String, TypeIdIndex>,
    /// Types that exist in `base` but were removed/invalidated post-load.
    removed: std::collections::HashSet<String>,
    base: Option<Arc<IdIndexBase>>,
}

/// Either a borrow into the overlay's `TypeIdIndex` or a slice of the base mmap.
pub enum IdIndexRef<'a> {
    Overlay(&'a TypeIdIndex),
    Base {
        base: &'a IdIndexBase,
        name: &'a str,
    },
}

impl<'a> IdIndexRef<'a> {
    pub fn get(&self, id: &Value) -> Option<NodeIndex> {
        match self {
            IdIndexRef::Overlay(idx) => idx.get(id),
            IdIndexRef::Base { base, name } => base.lookup(name, id),
        }
    }

    /// Iterate (Value, NodeIndex) pairs. For the base path this materializes
    /// per call — used only by save and by the rare `iter()` call sites.
    pub fn iter(&self) -> Box<dyn Iterator<Item = (Value, NodeIndex)> + '_> {
        match self {
            IdIndexRef::Overlay(idx) => idx.iter(),
            IdIndexRef::Base { base, name } => match base.materialize(name) {
                Some(materialized) => Box::new(materialized.iter().collect::<Vec<_>>().into_iter()),
                None => Box::new(std::iter::empty()),
            },
        }
    }
}

impl IdIndexStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_base(base: IdIndexBase) -> Self {
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

    pub fn get(&self, name: &str) -> Option<IdIndexRef<'_>> {
        if let Some(idx) = self.overlay.get(name) {
            return Some(IdIndexRef::Overlay(idx));
        }
        if self.removed.contains(name) {
            return None;
        }
        let base = self.base.as_deref()?;
        if base.contains(name) {
            // Resolve to the base's owned String key so the returned ref
            // does not borrow `name` (caller's string slice).
            let base_name = base.dir.get_key_value(name).map(|(k, _)| k.as_str())?;
            Some(IdIndexRef::Base {
                base,
                name: base_name,
            })
        } else {
            None
        }
    }

    pub fn insert(&mut self, name: String, idx: TypeIdIndex) {
        self.removed.remove(&name);
        self.overlay.insert(name, idx);
    }

    pub fn remove(&mut self, name: &str) -> Option<TypeIdIndex> {
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

    /// Iterate IdIndexRefs for every live entry (overlay first, then base).
    pub fn values(&self) -> impl Iterator<Item = IdIndexRef<'_>> {
        self.iter().map(|(_, v)| v)
    }

    /// Iterate `(name, IdIndexRef)` for every live entry.
    pub fn iter(&self) -> impl Iterator<Item = (&str, IdIndexRef<'_>)> {
        // Materialize the union deterministically: overlay entries first,
        // then base entries that aren't shadowed.
        let overlay_pairs: Vec<(&str, IdIndexRef<'_>)> = self
            .overlay
            .iter()
            .map(|(k, v)| (k.as_str(), IdIndexRef::Overlay(v)))
            .collect();
        let base_pairs: Vec<(&str, IdIndexRef<'_>)> = match self.base.as_deref() {
            Some(base) => base
                .dir
                .keys()
                .filter(|k| {
                    !self.overlay.contains_key(k.as_str()) && !self.removed.contains(k.as_str())
                })
                .map(move |k| {
                    (
                        k.as_str(),
                        IdIndexRef::Base {
                            base,
                            name: k.as_str(),
                        },
                    )
                })
                .collect(),
            None => Vec::new(),
        };
        overlay_pairs.into_iter().chain(base_pairs)
    }

    /// HashMap-`entry`-shaped accessor that materializes any base entry into
    /// the overlay before returning it (or default-constructs a new one).
    /// The returned entry is owned by the overlay; subsequent reads see it
    /// in preference to the (now superseded) base entry.
    pub fn entry_or_default(&mut self, name: String) -> &mut TypeIdIndex {
        if !self.overlay.contains_key(&name) && !self.removed.contains(&name) {
            if let Some(base) = self.base.as_deref() {
                if let Some(materialized) = base.materialize(&name) {
                    self.overlay.insert(name.clone(), materialized);
                }
            }
        }
        self.removed.remove(&name);
        self.overlay.entry(name).or_default()
    }

    /// Replace the entire store with a fresh HashMap (used by load fallback
    /// for legacy `.bin.zst`-only graphs and by `reindex()`).
    pub fn replace_with(&mut self, map: HashMap<String, TypeIdIndex>) {
        self.overlay = map;
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
        Value::String(s) => strip_prefix_to_u32(s),
        _ => None,
    }
}

/// Same as `schema::strip_prefix_to_u32` but local to avoid an extra
/// `pub(crate)` re-export.
#[inline]
fn strip_prefix_to_u32(s: &str) -> Option<u32> {
    let digit_start = s.bytes().position(|b| b.is_ascii_digit())?;
    if digit_start == 0 {
        return None;
    }
    let prefix = &s.as_bytes()[..digit_start];
    if !prefix.iter().all(|b| b.is_ascii_alphabetic()) {
        return None;
    }
    s[digit_start..].parse::<u32>().ok()
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
    let mut entries: Vec<(u64, String, TypeIdIndex)> = Vec::new();
    let mut interner_clone = interner.clone();
    for (name, view) in store.iter() {
        let key = interner_clone.get_or_intern(name).as_u64();
        let materialized = match &view {
            IdIndexRef::Overlay(idx) => (*idx).clone(),
            IdIndexRef::Base { base, name } => match base.materialize(name) {
                Some(m) => m,
                None => continue,
            },
        };
        entries.push((key, name.to_string(), materialized));
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
