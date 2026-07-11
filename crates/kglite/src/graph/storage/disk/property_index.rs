//! Persistent per-type string property index for `DiskGraph`.
//!
//! On-disk layout (mirrors the connection-type inverted index pattern):
//!
//! ```text
//! property_index_v2_{sha256}_meta.bin      // versioned exact identity + lengths
//! property_index_v2_{sha256}_keys.bin      // u8 concatenated sorted UTF-8 bytes
//! property_index_v2_{sha256}_offsets.bin   // u64 cumulative byte offsets (count + 1 entries)
//! property_index_v2_{sha256}_ids.bin       // u32 NodeIndex parallel to keys (count entries)
//! ```
//!
//! `meta.bin` is explicit because `MmapOrVec::mapped(...)` pads the
//! file to a minimum of 64 elements; the file size alone cannot recover
//! the logical count.
//!
//! Keys are sorted lexicographically; duplicates are adjacent and keyed
//! to their own `NodeIndex`. A single binary search yields the left and
//! right bounds of a run, so equality lookup is O(log N + k) where k is
//! the number of matches. The same layout supports prefix lookup by
//! range-scanning between `lower_bound(prefix)` and
//! `lower_bound(next_prefix)`.
//!
//! Restricted to `TypedColumn::Str` columns — numeric equality is a
//! follow-up.

use crate::graph::schema::InternedKey;
use crate::graph::storage::mapped::mmap_vec::MmapBytes;
use petgraph::graph::NodeIndex;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

/// Filename prefix for per-type on-disk property index files.
const FILE_PREFIX: &str = "property_index_";
const V2_FILE_PREFIX: &str = "property_index_v2_";

/// Filename prefix for cross-type "global" property index files.
/// A global index stores `(value, NodeIndex)` pairs for every live
/// node in the graph whose property resolves to a non-empty string,
/// regardless of node type. Used by untyped patterns like
/// `MATCH (n {label: 'X'})` and by the `search(text)` helper.
const GLOBAL_PREFIX: &str = "global_index_";
const V2_GLOBAL_PREFIX: &str = "global_index_v2_";
const META_MAGIC: &[u8; 8] = b"KGPIDX2\0";

#[derive(Clone, Copy)]
enum IndexIdentity<'a> {
    Typed {
        node_type: &'a str,
        property: &'a str,
    },
    Global {
        property: &'a str,
    },
}

impl<'a> IndexIdentity<'a> {
    fn kind(self) -> u8 {
        match self {
            Self::Typed { .. } => 0,
            Self::Global { .. } => 1,
        }
    }

    fn node_type(self) -> &'a str {
        match self {
            Self::Typed { node_type, .. } => node_type,
            Self::Global { .. } => "",
        }
    }

    fn property(self) -> &'a str {
        match self {
            Self::Typed { property, .. } | Self::Global { property } => property,
        }
    }
}

struct IndexMeta {
    count: usize,
    keys_len: usize,
    kind: u8,
    node_type: String,
    property: String,
}

/// A single `(node_type, property)` string index, backed by three
/// mmap'd files.
pub struct PropertyIndex {
    keys: MmapBytes,
    offsets: MmapBytes,
    ids: MmapBytes,
    count: usize,
}

/// Sanitise a node-type or property identifier for inclusion in a
/// filename. Strips anything outside `[A-Za-z0-9_-]` to `_` so users
/// with exotic type names don't confuse the path layer.
fn legacy_sanitise(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn digest_stem(prefix: &str, identity: IndexIdentity<'_>) -> String {
    let mut digest = Sha256::new();
    digest.update(b"kglite-property-index-v2\0");
    digest.update([identity.kind()]);
    digest.update((identity.node_type().len() as u64).to_le_bytes());
    digest.update(identity.node_type().as_bytes());
    digest.update((identity.property().len() as u64).to_le_bytes());
    digest.update(identity.property().as_bytes());
    let digest = digest.finalize();
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write;
        write!(&mut hex, "{byte:02x}").unwrap();
    }
    format!("{prefix}{hex}")
}

fn paths_for_stem(data_dir: &Path, stem: &str) -> (PathBuf, PathBuf, PathBuf, PathBuf) {
    (
        data_dir.join(format!("{stem}_meta.bin")),
        data_dir.join(format!("{stem}_keys.bin")),
        data_dir.join(format!("{stem}_offsets.bin")),
        data_dir.join(format!("{stem}_ids.bin")),
    )
}

/// Four file paths for `(node_type, property)` under `data_dir`:
/// returns `(meta, keys, offsets, ids)`.
pub fn file_paths(
    data_dir: &Path,
    node_type: &str,
    property: &str,
) -> (PathBuf, PathBuf, PathBuf, PathBuf) {
    let identity = IndexIdentity::Typed {
        node_type,
        property,
    };
    paths_for_stem(data_dir, &digest_stem(V2_FILE_PREFIX, identity))
}

fn legacy_file_paths(
    data_dir: &Path,
    node_type: &str,
    property: &str,
) -> (PathBuf, PathBuf, PathBuf, PathBuf) {
    let stem = format!(
        "{}{}_{}",
        FILE_PREFIX,
        legacy_sanitise(node_type),
        legacy_sanitise(property)
    );
    paths_for_stem(data_dir, &stem)
}

pub(crate) fn removal_paths(data_dir: &Path, node_type: &str, property: &str) -> Vec<PathBuf> {
    let current = file_paths(data_dir, node_type, property);
    let legacy = legacy_file_paths(data_dir, node_type, property);
    vec![
        current.0, current.1, current.2, current.3, legacy.0, legacy.1, legacy.2, legacy.3,
    ]
}

/// Four file paths for a cross-type global index keyed by `property`.
/// Returns `(meta, keys, offsets, ids)`.
pub fn global_file_paths(data_dir: &Path, property: &str) -> (PathBuf, PathBuf, PathBuf, PathBuf) {
    let identity = IndexIdentity::Global { property };
    paths_for_stem(data_dir, &digest_stem(V2_GLOBAL_PREFIX, identity))
}

fn legacy_global_file_paths(
    data_dir: &Path,
    property: &str,
) -> (PathBuf, PathBuf, PathBuf, PathBuf) {
    paths_for_stem(
        data_dir,
        &format!("{}{}", GLOBAL_PREFIX, legacy_sanitise(property)),
    )
}

/// Scan `data_dir` for validated v2 metadata and return the exact
/// `(node_type, property)` pairs they cover. Legacy filenames are omitted:
/// sanitisation and underscore splitting destroyed their exact identity.
///
/// Used by `build_single_segment_manifest` to discover which indexes
/// exist in a segment and populate `SegmentSummary.indexed_prop_ranges`.
pub fn scan_data_dir(data_dir: &Path) -> std::io::Result<Vec<(String, String)>> {
    let entries = match fs::read_dir(data_dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error),
    };
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for entry in entries {
        let entry = entry?;
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if name.starts_with(V2_FILE_PREFIX) && name.ends_with("_meta.bin") {
            if !entry.file_type()?.is_file() {
                return Err(invalid_index("v2 metadata path is not a regular file"));
            }
            let meta = read_v2_meta(&entry.path())?;
            if meta.kind != 0 {
                return Err(invalid_index("typed filename contains global metadata"));
            }
            let identity = IndexIdentity::Typed {
                node_type: &meta.node_type,
                property: &meta.property,
            };
            let expected = format!("{}_meta.bin", digest_stem(V2_FILE_PREFIX, identity));
            if name != expected {
                return Err(invalid_index(
                    "v2 metadata digest does not match its filename",
                ));
            }
            let pair = (meta.node_type, meta.property);
            if seen.insert(pair.clone()) {
                out.push(pair);
            }
        }
    }
    out.sort();
    Ok(out)
}

/// Like [`scan_data_dir`] but returns exact identities as
/// `(type_hash, prop_hash)` tuples in deterministic order.
pub fn scan_segment_hashes(segment_dir: &Path) -> std::io::Result<Vec<(u64, u64)>> {
    Ok(scan_data_dir(segment_dir)?
        .into_iter()
        .map(|(t, p)| {
            (
                InternedKey::from_str(&t).as_u64(),
                InternedKey::from_str(&p).as_u64(),
            )
        })
        .collect())
}

/// Validate every recognized v2 typed/global bundle in a segment. Legacy
/// files remain readable by exact-request fallback but cannot be bulk-
/// validated because their filenames destroyed the original identity.
pub fn validate_v2_bundles(data_dir: &Path) -> std::io::Result<()> {
    for (node_type, property) in scan_data_dir(data_dir)? {
        let paths = file_paths(data_dir, &node_type, &property);
        let identity = IndexIdentity::Typed {
            node_type: &node_type,
            property: &property,
        };
        if PropertyIndex::open_at(&paths.0, &paths.1, &paths.2, &paths.3, Some(identity))?.is_none()
        {
            return Err(invalid_index("typed v2 bundle is incomplete"));
        }
    }
    let entries = match fs::read_dir(data_dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };
    for entry in entries {
        let entry = entry?;
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if !name.starts_with(V2_GLOBAL_PREFIX) || !name.ends_with("_meta.bin") {
            continue;
        }
        if !entry.file_type()?.is_file() {
            return Err(invalid_index(
                "global v2 metadata path is not a regular file",
            ));
        }
        let meta = read_v2_meta(&entry.path())?;
        if meta.kind != 1 {
            return Err(invalid_index("global filename contains typed metadata"));
        }
        let identity = IndexIdentity::Global {
            property: &meta.property,
        };
        let expected = format!("{}_meta.bin", digest_stem(V2_GLOBAL_PREFIX, identity));
        if name != expected {
            return Err(invalid_index(
                "global v2 metadata digest does not match its filename",
            ));
        }
        let paths = global_file_paths(data_dir, &meta.property);
        if PropertyIndex::open_at(&paths.0, &paths.1, &paths.2, &paths.3, Some(identity))?.is_none()
        {
            return Err(invalid_index("global v2 bundle is incomplete"));
        }
    }
    Ok(())
}

/// Write versioned identity metadata last, after every payload file.
fn write_meta(
    path: &Path,
    count: usize,
    keys_len: usize,
    identity: IndexIdentity<'_>,
) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = fs::File::create(path)?;
    f.write_all(META_MAGIC)?;
    f.write_all(&[identity.kind()])?;
    f.write_all(&[0; 7])?;
    f.write_all(&(count as u64).to_le_bytes())?;
    f.write_all(&(keys_len as u64).to_le_bytes())?;
    let node_type = identity.node_type().as_bytes();
    let property = identity.property().as_bytes();
    f.write_all(
        &u32::try_from(node_type.len())
            .map_err(|_| invalid_index("node type name is too long"))?
            .to_le_bytes(),
    )?;
    f.write_all(
        &u32::try_from(property.len())
            .map_err(|_| invalid_index("property name is too long"))?
            .to_le_bytes(),
    )?;
    f.write_all(node_type)?;
    f.write_all(property)?;
    Ok(())
}

fn read_legacy_meta(path: &Path) -> std::io::Result<(usize, usize)> {
    let bytes = fs::read(path)?;
    if bytes.len() != 16 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "meta file too small",
        ));
    }
    let count = usize::try_from(u64::from_le_bytes(bytes[0..8].try_into().unwrap()))
        .map_err(|_| invalid_index("count exceeds usize"))?;
    let keys_len = usize::try_from(u64::from_le_bytes(bytes[8..16].try_into().unwrap()))
        .map_err(|_| invalid_index("keys length exceeds usize"))?;
    Ok((count, keys_len))
}

fn read_v2_meta(path: &Path) -> std::io::Result<IndexMeta> {
    const HEADER_LEN: usize = 40;
    const MAX_META_LEN: u64 = 1024 * 1024;
    if fs::metadata(path)?.len() > MAX_META_LEN {
        return Err(invalid_index("v2 metadata exceeds the size limit"));
    }
    let bytes = fs::read(path)?;
    if bytes.len() < HEADER_LEN || &bytes[..8] != META_MAGIC {
        return Err(invalid_index("v2 metadata header is invalid"));
    }
    let kind = bytes[8];
    if kind > 1 || bytes[9..16].iter().any(|byte| *byte != 0) {
        return Err(invalid_index(
            "v2 metadata kind or reserved bytes are invalid",
        ));
    }
    let count = usize::try_from(u64::from_le_bytes(bytes[16..24].try_into().unwrap()))
        .map_err(|_| invalid_index("count exceeds usize"))?;
    let keys_len = usize::try_from(u64::from_le_bytes(bytes[24..32].try_into().unwrap()))
        .map_err(|_| invalid_index("keys length exceeds usize"))?;
    let type_len = u32::from_le_bytes(bytes[32..36].try_into().unwrap()) as usize;
    let prop_len = u32::from_le_bytes(bytes[36..40].try_into().unwrap()) as usize;
    let expected_len = HEADER_LEN
        .checked_add(type_len)
        .and_then(|len| len.checked_add(prop_len))
        .ok_or_else(|| invalid_index("v2 identity length overflow"))?;
    if bytes.len() != expected_len {
        return Err(invalid_index("v2 metadata identity length is inconsistent"));
    }
    let node_type = std::str::from_utf8(&bytes[HEADER_LEN..HEADER_LEN + type_len])
        .map_err(|_| invalid_index("node type identity is not UTF-8"))?
        .to_string();
    let property = std::str::from_utf8(&bytes[HEADER_LEN + type_len..])
        .map_err(|_| invalid_index("property identity is not UTF-8"))?
        .to_string();
    if property.is_empty()
        || (kind == 0 && node_type.is_empty())
        || (kind == 1 && !node_type.is_empty())
    {
        return Err(invalid_index("v2 metadata identity is invalid"));
    }
    Ok(IndexMeta {
        count,
        keys_len,
        kind,
        node_type,
        property,
    })
}

fn validate_meta_identity(meta: &IndexMeta, identity: IndexIdentity<'_>) -> std::io::Result<()> {
    if meta.kind != identity.kind()
        || meta.node_type != identity.node_type()
        || meta.property != identity.property()
    {
        return Err(invalid_index(
            "v2 metadata identity does not match the requested index",
        ));
    }
    Ok(())
}

fn invalid_index(message: &str) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("invalid property index: {message}"),
    )
}

fn read_le_u64(bytes: &MmapBytes, index: usize) -> Option<u64> {
    let start = index.checked_mul(8)?;
    Some(u64::from_le_bytes(
        bytes.slice(start, start.checked_add(8)?).try_into().ok()?,
    ))
}

fn read_le_u32(bytes: &MmapBytes, index: usize) -> Option<u32> {
    let start = index.checked_mul(4)?;
    Some(u32::from_le_bytes(
        bytes.slice(start, start.checked_add(4)?).try_into().ok()?,
    ))
}

impl PropertyIndex {
    /// Number of indexed `(key, node_id)` entries.
    #[allow(dead_code)] // Test-only.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Borrow the i-th key as a byte slice.
    fn key_at(&self, i: usize) -> &[u8] {
        let start = read_le_u64(&self.offsets, i).unwrap() as usize;
        let end = read_le_u64(&self.offsets, i + 1).unwrap() as usize;
        self.keys.slice(start, end)
    }

    /// Return the left-most index `i` such that `key_at(i) >= target`.
    fn lower_bound(&self, target: &[u8]) -> usize {
        let (mut lo, mut hi) = (0usize, self.count);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.key_at(mid) < target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// Exact-match lookup: return all `NodeIndex` values whose key equals
    /// `value`. Duplicates are returned in the order they were emitted
    /// by the build pass (ascending `NodeIndex` within each key —
    /// produced by a stable sort).
    pub fn lookup_eq_str(&self, value: &str) -> Vec<NodeIndex> {
        let target = value.as_bytes();
        let start = self.lower_bound(target);
        if start >= self.count || self.key_at(start) != target {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut i = start;
        while i < self.count && self.key_at(i) == target {
            out.push(NodeIndex::new(read_le_u32(&self.ids, i).unwrap() as usize));
            i += 1;
        }
        out
    }

    /// Prefix lookup: return `NodeIndex` values whose key starts with
    /// `prefix`, capped at `limit` results. Traverses a contiguous run
    /// of the sorted index, so work is bounded by the number of matches.
    pub fn lookup_prefix_str(&self, prefix: &str, limit: usize) -> Vec<NodeIndex> {
        if limit == 0 {
            return Vec::new();
        }
        let target = prefix.as_bytes();
        let start = self.lower_bound(target);
        let mut out = Vec::with_capacity(limit.min(16));
        let mut i = start;
        while i < self.count && out.len() < limit {
            let key = self.key_at(i);
            if !key.starts_with(target) {
                break;
            }
            out.push(NodeIndex::new(read_le_u32(&self.ids, i).unwrap() as usize));
            i += 1;
        }
        out
    }

    /// Build a fresh index from a collection of `(key, node_index)`
    /// pairs, serialise to `data_dir`, and return an opened handle.
    ///
    /// `entries` may be in any order; this routine sorts them by
    /// `(key, node_index)` for a deterministic layout.
    pub fn build(
        data_dir: &Path,
        node_type: &str,
        property: &str,
        entries: Vec<(String, u32)>,
    ) -> std::io::Result<Self> {
        let paths = file_paths(data_dir, node_type, property);
        Self::build_at(
            &paths.0,
            &paths.1,
            &paths.2,
            &paths.3,
            entries,
            IndexIdentity::Typed {
                node_type,
                property,
            },
        )
    }

    /// Build a cross-type global index keyed by `property`. Files are
    /// written with the `global_index_` prefix. Lookup semantics mirror
    /// the per-type variant; the caller decides whether to pair the
    /// resulting `NodeIndex` with its type by reading
    /// `node_type_of(idx)` afterward.
    pub fn build_global(
        data_dir: &Path,
        property: &str,
        entries: Vec<(String, u32)>,
    ) -> std::io::Result<Self> {
        let paths = global_file_paths(data_dir, property);
        Self::build_at(
            &paths.0,
            &paths.1,
            &paths.2,
            &paths.3,
            entries,
            IndexIdentity::Global { property },
        )
    }

    /// Low-level build: write the four files at the supplied paths.
    /// Shared by [`build`] and [`build_global`].
    fn build_at(
        meta_path: &Path,
        keys_path: &Path,
        offsets_path: &Path,
        ids_path: &Path,
        mut entries: Vec<(String, u32)>,
        identity: IndexIdentity<'_>,
    ) -> std::io::Result<Self> {
        entries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        let total_bytes: usize = entries.iter().map(|(k, _)| k.len()).sum();
        let count = entries.len();

        let mut keys_raw = Vec::with_capacity(total_bytes);
        let mut offsets_raw = Vec::with_capacity((count + 1) * 8);
        let mut ids_raw = Vec::with_capacity(count * 4);
        let mut offset = 0u64;
        for (key, id) in &entries {
            offsets_raw.extend_from_slice(&offset.to_le_bytes());
            keys_raw.extend_from_slice(key.as_bytes());
            ids_raw.extend_from_slice(&id.to_le_bytes());
            offset = offset
                .checked_add(key.len() as u64)
                .ok_or_else(|| invalid_index("key byte count overflow"))?;
        }
        offsets_raw.extend_from_slice(&offset.to_le_bytes());

        // Invalidate any prior index before replacing its backing files. If a
        // later write fails, readers must not combine stale metadata with a
        // partial new payload and mistake it for a valid index.
        match fs::remove_file(meta_path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
        fs::write(keys_path, &keys_raw)?;
        fs::write(offsets_path, &offsets_raw)?;
        fs::write(ids_path, &ids_raw)?;
        write_meta(meta_path, count, total_bytes, identity)?;
        Self::open_at(meta_path, keys_path, offsets_path, ids_path, Some(identity))?
            .ok_or_else(|| invalid_index("freshly written property index files disappeared"))
    }

    /// Re-open a previously built per-type index. Returns `Ok(None)`
    /// if any file is missing.
    pub fn open(data_dir: &Path, node_type: &str, property: &str) -> std::io::Result<Option<Self>> {
        let paths = file_paths(data_dir, node_type, property);
        let identity = IndexIdentity::Typed {
            node_type,
            property,
        };
        if let Some(index) = Self::open_at(&paths.0, &paths.1, &paths.2, &paths.3, Some(identity))?
        {
            return Ok(Some(index));
        }
        let legacy = legacy_file_paths(data_dir, node_type, property);
        Self::open_at(&legacy.0, &legacy.1, &legacy.2, &legacy.3, None)
    }

    /// Re-open a cross-type global index. Returns `Ok(None)` if any
    /// file is missing.
    pub fn open_global(data_dir: &Path, property: &str) -> std::io::Result<Option<Self>> {
        let paths = global_file_paths(data_dir, property);
        let identity = IndexIdentity::Global { property };
        if let Some(index) = Self::open_at(&paths.0, &paths.1, &paths.2, &paths.3, Some(identity))?
        {
            return Ok(Some(index));
        }
        let legacy = legacy_global_file_paths(data_dir, property);
        Self::open_at(&legacy.0, &legacy.1, &legacy.2, &legacy.3, None)
    }

    fn open_at(
        meta_path: &Path,
        keys_path: &Path,
        offsets_path: &Path,
        ids_path: &Path,
        identity: Option<IndexIdentity<'_>>,
    ) -> std::io::Result<Option<Self>> {
        if !meta_path.exists()
            || !keys_path.exists()
            || !offsets_path.exists()
            || !ids_path.exists()
        {
            return Ok(None);
        }
        if identity.is_some() {
            for path in [meta_path, keys_path, offsets_path, ids_path] {
                if !fs::symlink_metadata(path)?.file_type().is_file() {
                    return Err(invalid_index("v2 bundle member is not a regular file"));
                }
            }
        }
        let (count, keys_len) = if let Some(expected) = identity {
            let meta = read_v2_meta(meta_path)?;
            validate_meta_identity(&meta, expected)?;
            (meta.count, meta.keys_len)
        } else {
            read_legacy_meta(meta_path)?
        };
        let offsets_len = count
            .checked_add(1)
            .and_then(|n| n.checked_mul(8))
            .ok_or_else(|| invalid_index("offset array size overflow"))?;
        let ids_len = count
            .checked_mul(4)
            .ok_or_else(|| invalid_index("id array size overflow"))?;
        let offsets_raw = fs::read(offsets_path)?;
        let ids_raw = fs::read(ids_path)?;
        let keys_raw = fs::read(keys_path)?;
        let invalid_lengths = if identity.is_some() {
            offsets_raw.len() != offsets_len
                || ids_raw.len() != ids_len
                || keys_raw.len() != keys_len
        } else {
            offsets_raw.len() < offsets_len || ids_raw.len() < ids_len || keys_raw.len() < keys_len
        };
        if invalid_lengths {
            return Err(invalid_index("one or more index files are truncated"));
        }
        let mut previous_offset = 0u64;
        let mut previous_pair: Option<(&[u8], u32)> = None;
        for index in 0..=count {
            let start = index * 8;
            let offset = u64::from_le_bytes(offsets_raw[start..start + 8].try_into().unwrap());
            if (index == 0 && offset != 0) || offset < previous_offset || offset > keys_len as u64 {
                return Err(invalid_index("offsets are not monotonic or contained"));
            }
            if index < count {
                let next =
                    u64::from_le_bytes(offsets_raw[start + 8..start + 16].try_into().unwrap());
                if next < offset || next > keys_len as u64 {
                    return Err(invalid_index("key range is outside keys file"));
                }
                let key = &keys_raw[offset as usize..next as usize];
                std::str::from_utf8(key)
                    .map_err(|_| invalid_index("key bytes contain invalid UTF-8"))?;
                let id_start = index * 4;
                let id = u32::from_le_bytes(ids_raw[id_start..id_start + 4].try_into().unwrap());
                if previous_pair.is_some_and(|(prior_key, prior_id)| {
                    key < prior_key || (key == prior_key && id <= prior_id)
                }) {
                    return Err(invalid_index("(key, id) pairs are not strictly sorted"));
                }
                previous_pair = Some((key, id));
            }
            previous_offset = offset;
        }
        if previous_offset != keys_len as u64 {
            return Err(invalid_index("final offset does not equal keys length"));
        }
        let keys = MmapBytes::load_mapped(keys_path, keys_len)?;
        let offsets = MmapBytes::load_mapped(offsets_path, offsets_len)?;
        let ids = MmapBytes::load_mapped(ids_path, ids_len)?;
        Ok(Some(PropertyIndex {
            keys,
            offsets,
            ids,
            count,
        }))
    }

    /// Delete the on-disk files for this index. Used in tests covering
    /// `drop_index` for the disk path.
    #[allow(dead_code)] // Test-only.
    pub fn remove_files(data_dir: &Path, node_type: &str, property: &str) -> std::io::Result<()> {
        for p in removal_paths(data_dir, node_type, property) {
            if p.exists() {
                fs::remove_file(&p)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "kglite_prop_idx_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn equality_lookup_finds_single_match() {
        let dir = tmp_dir();
        let idx = PropertyIndex::build(
            &dir,
            "Human",
            "label",
            vec![
                ("Alice".into(), 1),
                ("Bob".into(), 2),
                ("Charlie".into(), 3),
            ],
        )
        .unwrap();
        assert_eq!(idx.lookup_eq_str("Bob"), vec![NodeIndex::new(2)]);
        assert_eq!(idx.lookup_eq_str("Alice"), vec![NodeIndex::new(1)]);
        assert!(idx.lookup_eq_str("Missing").is_empty());
    }

    #[test]
    fn equality_lookup_returns_all_duplicates() {
        let dir = tmp_dir();
        let idx = PropertyIndex::build(
            &dir,
            "Human",
            "label",
            vec![
                ("Alice".into(), 1),
                ("Alice".into(), 7),
                ("Alice".into(), 4),
                ("Bob".into(), 2),
            ],
        )
        .unwrap();
        let hits = idx.lookup_eq_str("Alice");
        assert_eq!(
            hits,
            vec![NodeIndex::new(1), NodeIndex::new(4), NodeIndex::new(7)]
        );
    }

    #[test]
    fn prefix_lookup_respects_limit_and_sort_order() {
        let dir = tmp_dir();
        let idx = PropertyIndex::build(
            &dir,
            "Human",
            "label",
            vec![
                ("Oslo".into(), 10),
                ("Ottawa".into(), 11),
                ("Oxford".into(), 12),
                ("Paris".into(), 13),
            ],
        )
        .unwrap();
        let hits = idx.lookup_prefix_str("O", 10);
        assert_eq!(
            hits,
            vec![NodeIndex::new(10), NodeIndex::new(11), NodeIndex::new(12)]
        );
        // Limit 2 returns the first two only.
        let hits = idx.lookup_prefix_str("O", 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0], NodeIndex::new(10));
    }

    #[test]
    fn persistence_roundtrip_via_open() {
        let dir = tmp_dir();
        {
            let _ = PropertyIndex::build(
                &dir,
                "Human",
                "label",
                vec![("Alice".into(), 1), ("Bob".into(), 2)],
            )
            .unwrap();
        } // drop: flush to disk
        let reopened = PropertyIndex::open(&dir, "Human", "label")
            .unwrap()
            .expect("index files should be present");
        assert_eq!(reopened.lookup_eq_str("Bob"), vec![NodeIndex::new(2)]);
        assert_eq!(reopened.len(), 2);
    }

    #[test]
    fn v2_paths_keep_former_collision_pairs_independent() {
        let dir = tmp_dir();
        let cases = [
            ("a_b", "c"),
            ("a", "b_c"),
            ("a.b", "c"),
            ("a/b", "c"),
            ("Å", "naïve/property"),
            ("Case", "name"),
            ("case", "name"),
        ];
        let mut meta_paths = HashSet::new();
        for (index, (node_type, property)) in cases.into_iter().enumerate() {
            let paths = file_paths(&dir, node_type, property);
            assert!(paths.0.file_name().unwrap().len() < 128);
            assert!(meta_paths.insert(paths.0.clone()));
            PropertyIndex::build(
                &dir,
                node_type,
                property,
                vec![(format!("value-{index}"), index as u32)],
            )
            .unwrap();
        }
        for (index, (node_type, property)) in cases.into_iter().enumerate() {
            let opened = PropertyIndex::open(&dir, node_type, property)
                .unwrap()
                .unwrap();
            assert_eq!(
                opened.lookup_eq_str(&format!("value-{index}")),
                vec![NodeIndex::new(index)]
            );
        }
        let scanned: HashSet<_> = scan_data_dir(&dir).unwrap().into_iter().collect();
        assert_eq!(
            scanned,
            cases
                .into_iter()
                .map(|(t, p)| (t.into(), p.into()))
                .collect()
        );
    }

    #[test]
    fn global_v2_paths_do_not_sanitise_collisions() {
        let dir = tmp_dir();
        PropertyIndex::build_global(&dir, "a.b", vec![("dot".into(), 1)]).unwrap();
        PropertyIndex::build_global(&dir, "a/b", vec![("slash".into(), 2)]).unwrap();
        assert_ne!(
            global_file_paths(&dir, "a.b").0,
            global_file_paths(&dir, "a/b").0
        );
        assert_eq!(
            PropertyIndex::open_global(&dir, "a.b")
                .unwrap()
                .unwrap()
                .lookup_eq_str("dot"),
            vec![NodeIndex::new(1)]
        );
        assert_eq!(
            PropertyIndex::open_global(&dir, "a/b")
                .unwrap()
                .unwrap()
                .lookup_eq_str("slash"),
            vec![NodeIndex::new(2)]
        );
    }

    #[test]
    fn legacy_bundle_remains_readable_but_scanner_does_not_invent_identity() {
        let dir = tmp_dir();
        PropertyIndex::build(&dir, "Human", "label", vec![("Alice".into(), 7)]).unwrap();
        let current = file_paths(&dir, "Human", "label");
        let meta = read_v2_meta(&current.0).unwrap();
        let legacy = legacy_file_paths(&dir, "Human", "label");
        for (source, target) in [
            (&current.1, &legacy.1),
            (&current.2, &legacy.2),
            (&current.3, &legacy.3),
        ] {
            fs::rename(source, target).unwrap();
        }
        let legacy_meta = [
            (meta.count as u64).to_le_bytes(),
            (meta.keys_len as u64).to_le_bytes(),
        ]
        .concat();
        fs::write(&legacy.0, legacy_meta).unwrap();
        fs::remove_file(&current.0).unwrap();

        let opened = PropertyIndex::open(&dir, "Human", "label")
            .unwrap()
            .unwrap();
        assert_eq!(opened.lookup_eq_str("Alice"), vec![NodeIndex::new(7)]);
        assert!(scan_data_dir(&dir).unwrap().is_empty());
    }

    #[test]
    fn legacy_global_bundle_remains_readable_by_exact_request() {
        let dir = tmp_dir();
        PropertyIndex::build_global(&dir, "label", vec![("Alice".into(), 9)]).unwrap();
        let current = global_file_paths(&dir, "label");
        let meta = read_v2_meta(&current.0).unwrap();
        let legacy = legacy_global_file_paths(&dir, "label");
        for (source, target) in [
            (&current.1, &legacy.1),
            (&current.2, &legacy.2),
            (&current.3, &legacy.3),
        ] {
            fs::rename(source, target).unwrap();
        }
        fs::write(
            &legacy.0,
            [
                (meta.count as u64).to_le_bytes(),
                (meta.keys_len as u64).to_le_bytes(),
            ]
            .concat(),
        )
        .unwrap();
        fs::remove_file(&current.0).unwrap();
        assert_eq!(
            PropertyIndex::open_global(&dir, "label")
                .unwrap()
                .unwrap()
                .lookup_eq_str("Alice"),
            vec![NodeIndex::new(9)]
        );
    }

    #[test]
    fn v2_metadata_identity_and_digest_are_validated() {
        let dir = tmp_dir();
        PropertyIndex::build(&dir, "Human", "label", vec![("Alice".into(), 1)]).unwrap();
        let paths = file_paths(&dir, "Human", "label");
        let mut meta = fs::read(&paths.0).unwrap();
        let last = meta.last_mut().unwrap();
        *last = b'X';
        fs::write(&paths.0, meta).unwrap();
        assert_open_invalid(&dir);
        assert!(scan_data_dir(&dir).is_err());
    }

    #[test]
    fn v2_bundle_validation_rejects_missing_payload() {
        let dir = tmp_dir();
        PropertyIndex::build(&dir, "Human", "label", vec![("Alice".into(), 1)]).unwrap();
        let paths = file_paths(&dir, "Human", "label");
        fs::remove_file(paths.3).unwrap();
        assert!(validate_v2_bundles(&dir).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn v2_scanner_rejects_metadata_symlink() {
        use std::os::unix::fs::symlink;

        let dir = tmp_dir();
        PropertyIndex::build(&dir, "Human", "label", vec![("Alice".into(), 1)]).unwrap();
        let paths = file_paths(&dir, "Human", "label");
        let real = dir.join("real-meta.bin");
        fs::rename(&paths.0, &real).unwrap();
        symlink(&real, &paths.0).unwrap();
        assert!(scan_data_dir(&dir).is_err());
    }

    #[test]
    fn scan_data_dir_discovers_built_indexes() {
        let dir = tmp_dir();
        let _ = PropertyIndex::build(&dir, "Human", "label", vec![("A".into(), 1)]).unwrap();
        let _ = PropertyIndex::build(&dir, "Paper", "title", vec![("Z".into(), 2)]).unwrap();
        let mut pairs = scan_data_dir(&dir).unwrap();
        pairs.sort();
        assert_eq!(
            pairs,
            vec![
                ("Human".to_string(), "label".to_string()),
                ("Paper".to_string(), "title".to_string()),
            ]
        );
    }

    #[test]
    fn remove_files_cleans_up() {
        let dir = tmp_dir();
        let _ = PropertyIndex::build(&dir, "Human", "label", vec![("A".into(), 1)]).unwrap();
        PropertyIndex::remove_files(&dir, "Human", "label").unwrap();
        assert!(PropertyIndex::open(&dir, "Human", "label")
            .unwrap()
            .is_none());
    }

    #[test]
    fn empty_index_lookup_returns_empty() {
        let dir = tmp_dir();
        let idx = PropertyIndex::build(&dir, "Human", "label", Vec::new()).unwrap();
        assert!(idx.lookup_eq_str("anything").is_empty());
        assert!(idx.lookup_prefix_str("x", 10).is_empty());
    }

    #[test]
    fn scan_segment_hashes_returns_hashed_pairs() {
        // `tempfile::TempDir` avoids the nanosecond-timestamp collisions
        // that the legacy `tmp_dir()` helper can hit under parallel tests.
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path();
        let _ = PropertyIndex::build(dir, "Human", "label", vec![("A".into(), 1)]).unwrap();
        let _ = PropertyIndex::build(dir, "Paper", "title", vec![("Z".into(), 2)]).unwrap();

        let mut pairs = scan_segment_hashes(dir).unwrap();
        pairs.sort();
        let mut expected = vec![
            (
                InternedKey::from_str("Human").as_u64(),
                InternedKey::from_str("label").as_u64(),
            ),
            (
                InternedKey::from_str("Paper").as_u64(),
                InternedKey::from_str("title").as_u64(),
            ),
        ];
        expected.sort();
        assert_eq!(pairs, expected);
    }

    fn assert_open_invalid(dir: &Path) {
        let result = std::panic::catch_unwind(|| PropertyIndex::open(dir, "Human", "label"));
        match result.expect("invalid property index must not panic") {
            Err(error) => assert_eq!(error.kind(), std::io::ErrorKind::InvalidData),
            Ok(_) => panic!("invalid property index loaded successfully"),
        }
    }

    #[test]
    fn files_use_known_little_endian_integer_bytes() {
        let tmp = tempfile::TempDir::new().unwrap();
        PropertyIndex::build(
            tmp.path(),
            "Human",
            "label",
            vec![("A".into(), 0x7856_3412)],
        )
        .unwrap();
        let paths = file_paths(tmp.path(), "Human", "label");
        assert_eq!(
            fs::read(paths.2).unwrap(),
            [0u64.to_le_bytes(), 1u64.to_le_bytes()].concat()
        );
        assert_eq!(fs::read(paths.3).unwrap(), 0x7856_3412u32.to_le_bytes());
    }

    #[test]
    fn rejects_truncated_meta_and_offset_arrays() {
        let tmp = tempfile::TempDir::new().unwrap();
        PropertyIndex::build(tmp.path(), "Human", "label", vec![("A".into(), 1)]).unwrap();
        let paths = file_paths(tmp.path(), "Human", "label");
        fs::write(&paths.0, [0u8; 15]).unwrap();
        assert_open_invalid(tmp.path());

        PropertyIndex::build(tmp.path(), "Human", "label", vec![("A".into(), 1)]).unwrap();
        fs::write(&paths.2, [0u8; 8]).unwrap();
        assert_open_invalid(tmp.path());
    }

    #[test]
    fn rejects_non_monotonic_offsets_invalid_utf8_and_unsorted_pairs() {
        let tmp = tempfile::TempDir::new().unwrap();
        let entries = vec![("A".into(), 1), ("A".into(), 2)];
        PropertyIndex::build(tmp.path(), "Human", "label", entries.clone()).unwrap();
        let paths = file_paths(tmp.path(), "Human", "label");
        let bad_offsets = [0u64.to_le_bytes(), 2u64.to_le_bytes(), 1u64.to_le_bytes()].concat();
        fs::write(&paths.2, bad_offsets).unwrap();
        assert_open_invalid(tmp.path());

        PropertyIndex::build(tmp.path(), "Human", "label", entries.clone()).unwrap();
        fs::write(&paths.1, [0xff, b'A']).unwrap();
        assert_open_invalid(tmp.path());

        PropertyIndex::build(tmp.path(), "Human", "label", entries).unwrap();
        fs::write(&paths.3, [2u32.to_le_bytes(), 1u32.to_le_bytes()].concat()).unwrap();
        assert_open_invalid(tmp.path());
    }

    #[test]
    fn failed_rebuild_invalidates_stale_metadata() {
        let tmp = tempfile::TempDir::new().unwrap();
        PropertyIndex::build(tmp.path(), "Human", "label", vec![("old".into(), 1)]).unwrap();
        let paths = file_paths(tmp.path(), "Human", "label");

        fs::remove_file(&paths.3).unwrap();
        fs::create_dir(&paths.3).unwrap();
        let error = match PropertyIndex::build(
            tmp.path(),
            "Human",
            "label",
            vec![("replacement".into(), 2)],
        ) {
            Ok(_) => panic!("a directory at the ids path must fail the backing write"),
            Err(error) => error,
        };

        assert_ne!(error.kind(), std::io::ErrorKind::NotFound);
        assert!(
            !paths.0.exists(),
            "failed backing writes must not leave stale metadata published"
        );
        assert!(PropertyIndex::open(tmp.path(), "Human", "label")
            .unwrap()
            .is_none());
    }
}
