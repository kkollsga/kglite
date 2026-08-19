//! Fast-load sidecar binary formats for the `.kgl` / disk directory layout.
//!
//! Split out of `io/file.rs` for the production-source file cap, alongside
//! `metadata_sidecars` and `storage_mode` — these are the packed `*.bin.zst`
//! codecs (`type_indices`, `interner`, `id_indices`, `type_connectivity`,
//! `secondary_labels`) that replaced serde-rebuilt collections on the load
//! path. Each carries its own magic + version, and each reader returns
//! `Ok(None)`/`Ok(false)` for an unrecognised header so the caller can fall
//! back to rebuilding rather than failing a load.
//!
//! `use super::*` deliberately: these codecs are an extraction of the parent
//! module's body, and they lean on its shared helpers (`zstd_compress`,
//! `decode_disk_serde`, `invalid_data`) rather than re-deriving them.

use super::*;

// ─── type_indices.bin.zst (0.8.13 fast-load) ─────────────────────────────────
//
// Replaces a bincode-serialised `HashMap<String, Vec<NodeIndex>>` with a
// CSR-shaped packed binary keyed by interner hashes. On the 81 GB
// Wikidata graph this drops the load from bincode-rebuilt HashMap
// (88 k String keys + 124 M NodeIndex pushes spread across 88 k
// `Vec`s) to three packed slices + one exact-capacity HashMap build.
//
// Payload (pre-zstd):
//   [ 0.. 8]  magic       = b"KGLTIDX1"
//   [ 8..12]  version     = u32 LE (= 1)
//   [12..16]  num_types   = u32 LE
//   [16..24]  total_nodes = u64 LE
//   [24..24 + 8·num_types]             type_keys: [u64; num_types]
//   [next..next + 8·(num_types+1)]     offsets:   [u64; num_types+1]
//   [next..next + 4·total_nodes]       nodes:     [u32; total_nodes]
//
// `type_keys[i]` is `InternedKey::as_u64()` for the ith type name
// (sorted ascending by interner key for deterministic output).
// `nodes[offsets[i]..offsets[i+1]]` is the `NodeIndex` list for that
// type, stored as `NodeIndex::index() as u32` (graphs with >4 B
// nodes would need a bump here).

pub(crate) const TYPE_INDICES_MAGIC: &[u8; 8] = b"KGLTIDX1";
pub(crate) const TYPE_INDICES_VERSION: u32 = 1;

/// Reader for `type_indices.bin.zst` in the earlier flat-CSR format.
/// Returns `Ok(None)` if the payload does not start with the
/// `KGLTIDX1` magic so the caller can rebuild from node slots.
pub(crate) fn read_type_indices_bin(
    payload: &[u8],
    interner: &crate::graph::storage::interner::StringInterner,
) -> io::Result<Option<std::collections::HashMap<String, Vec<petgraph::graph::NodeIndex>>>> {
    if payload.len() < 24 || &payload[..8] != TYPE_INDICES_MAGIC {
        return Ok(None);
    }
    let version = u32::from_le_bytes(payload[8..12].try_into().unwrap());
    if version != TYPE_INDICES_VERSION {
        return Err(invalid_data("unsupported type_indices.bin.zst version"));
    }
    let num_types = u32::from_le_bytes(payload[12..16].try_into().unwrap()) as usize;
    let total_nodes = usize::try_from(u64::from_le_bytes(payload[16..24].try_into().unwrap()))
        .map_err(|_| invalid_data("type index node count exceeds usize"))?;

    let type_keys_offset = 24usize;
    let type_keys_bytes = 8usize
        .checked_mul(num_types)
        .ok_or_else(|| invalid_data("type index key directory size overflow"))?;
    let offsets_offset = type_keys_offset
        .checked_add(type_keys_bytes)
        .ok_or_else(|| invalid_data("type index offsets location overflow"))?;
    let offsets_bytes = num_types
        .checked_add(1)
        .and_then(|n| n.checked_mul(8))
        .ok_or_else(|| invalid_data("type index offset array size overflow"))?;
    let nodes_offset = offsets_offset
        .checked_add(offsets_bytes)
        .ok_or_else(|| invalid_data("type index nodes location overflow"))?;
    let node_bytes = total_nodes
        .checked_mul(4)
        .ok_or_else(|| invalid_data("type index node array size overflow"))?;
    let expected_len = nodes_offset
        .checked_add(node_bytes)
        .ok_or_else(|| invalid_data("type index total size overflow"))?;
    if payload.len() != expected_len {
        return Err(invalid_data(
            "type_indices.bin.zst size does not match header",
        ));
    }

    let mut out =
        std::collections::HashMap::<String, Vec<petgraph::graph::NodeIndex>>::with_capacity(
            num_types,
        );
    let first_offset = u64::from_le_bytes(
        payload[offsets_offset..offsets_offset + 8]
            .try_into()
            .unwrap(),
    );
    if first_offset != 0 {
        return Err(invalid_data("type index offsets must start at zero"));
    }
    let mut previous_type_key = None;
    let mut previous_offset = 0usize;
    for i in 0..num_types {
        let tkey_base = type_keys_offset + i * 8;
        let type_key = u64::from_le_bytes(payload[tkey_base..tkey_base + 8].try_into().unwrap());
        if previous_type_key.is_some_and(|previous| type_key <= previous) {
            return Err(invalid_data("type index keys are not strictly increasing"));
        }
        previous_type_key = Some(type_key);
        let off_base = offsets_offset + i * 8;
        let off_start = usize::try_from(u64::from_le_bytes(
            payload[off_base..off_base + 8].try_into().unwrap(),
        ))
        .map_err(|_| invalid_data("type index offset exceeds usize"))?;
        let off_end = usize::try_from(u64::from_le_bytes(
            payload[off_base + 8..off_base + 16].try_into().unwrap(),
        ))
        .map_err(|_| invalid_data("type index offset exceeds usize"))?;
        if off_start != previous_offset || off_end < off_start || off_end > total_nodes {
            return Err(invalid_data(
                "type index offsets are not monotonic or contained",
            ));
        }
        previous_offset = off_end;
        let name = interner
            .try_resolve(crate::graph::schema::InternedKey::from_u64(type_key))
            .ok_or_else(|| invalid_data("type index contains an unresolved type key"))?
            .to_string();
        let nodes_start = nodes_offset + off_start * 4;
        let nodes_end = nodes_offset + off_end * 4;
        let mut vec = Vec::with_capacity(off_end - off_start);
        let mut previous_node = None;
        for chunk in payload[nodes_start..nodes_end].chunks_exact(4) {
            let idx = u32::from_le_bytes(chunk.try_into().unwrap()) as usize;
            if previous_node.is_some_and(|previous| idx <= previous) {
                return Err(invalid_data(
                    "type index node ids are not strictly increasing",
                ));
            }
            previous_node = Some(idx);
            vec.push(petgraph::graph::NodeIndex::new(idx));
        }
        if out.insert(name, vec).is_some() {
            return Err(invalid_data("type index contains duplicate type names"));
        }
    }
    if previous_offset != total_nodes {
        return Err(invalid_data(
            "type index final offset disagrees with node count",
        ));
    }
    Ok(Some(out))
}

// ─── interner.bin.zst (0.8.13 fast-load) ─────────────────────────────────────
//
// Replaces `interner.json` (a `HashMap<String, String>` of
// hash-to-original) with a compact `Vec<String>` sidecar of the
// original strings, zstd-compressed. The hash is re-derived on load
// by `interner.get_or_intern` — FNV of the string is deterministic.
// Dropping the hash halves the on-disk size and eliminates JSON
// parse overhead.

pub(crate) fn write_interner_bin(dir: &std::path::Path, graph: &DirGraph) -> Result<(), String> {
    let originals: Vec<String> = graph.interner.iter().map(|(_, v)| v.to_string()).collect();
    let bytes = encode_disk_serde(&originals)
        .map_err(|e| format!("interner serialization failed: {}", e))?;
    let compressed = zstd::encode_all(bytes.as_slice(), 3)
        .map_err(|e| format!("interner compression failed: {}", e))?;
    std::fs::write(dir.join("interner.bin.zst"), compressed)
        .map_err(|e| format!("Failed to write interner.bin.zst: {}", e))?;
    Ok(())
}

pub(crate) fn read_interner_bin(dir: &std::path::Path, graph: &mut DirGraph) -> io::Result<bool> {
    let path = dir.join("interner.bin.zst");
    if !path.exists() {
        return Ok(false);
    }
    let compressed = std::fs::read(&path)?;
    let bytes = zstd_decompress(&compressed)?;
    let originals: Vec<String> = decode_disk_serde(&bytes, bytes.capacity() as u64)?;
    for s in &originals {
        graph
            .interner
            .try_get_or_intern(s)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    }
    Ok(true)
}

// ─── Retained flat-CSR id_indices.bin.zst reader ──────────────────────────
//
// Read-only fallback for graphs saved by 0.8.13–0.8.27. Fresh saves use
// the mmap-resident `id_indices.bin` raw layout from
// `storage/disk/id_index.rs::write_id_indices_bin`.

pub(crate) const ID_INDICES_MAGIC: &[u8; 8] = b"KGLIIDX1";
pub(crate) const ID_INDICES_VERSION: u32 = 1;

pub(crate) fn read_id_indices_bin(
    payload: &[u8],
    interner: &crate::graph::storage::interner::StringInterner,
) -> io::Result<Option<std::collections::HashMap<String, crate::graph::schema::TypeIdIndex>>> {
    use crate::graph::schema::TypeIdIndex;

    if payload.len() < 16 || &payload[..8] != ID_INDICES_MAGIC {
        return Ok(None);
    }
    let version = u32::from_le_bytes(payload[8..12].try_into().unwrap());
    if version != ID_INDICES_VERSION {
        return Err(invalid_data("unsupported id_indices.bin.zst version"));
    }
    let num_types = u32::from_le_bytes(payload[12..16].try_into().unwrap()) as usize;
    if num_types > (payload.len() - 16) / 24 {
        return Err(invalid_data(
            "id_indices.bin.zst directory count is truncated",
        ));
    }
    let mut out = std::collections::HashMap::<String, TypeIdIndex>::with_capacity(num_types);

    let mut cursor = 16usize;
    let mut previous_type_key = None;
    for _ in 0..num_types {
        let header_end = cursor
            .checked_add(24)
            .ok_or_else(|| invalid_data("id index block header overflow"))?;
        let header = payload
            .get(cursor..header_end)
            .ok_or_else(|| invalid_data("id_indices.bin.zst truncated at block header"))?;
        let type_key = u64::from_le_bytes(header[..8].try_into().unwrap());
        if previous_type_key.is_some_and(|previous| type_key <= previous) {
            return Err(invalid_data(
                "id index type keys are not strictly increasing",
            ));
        }
        previous_type_key = Some(type_key);
        let variant_tag = header[8];
        let num_entries = usize::try_from(u64::from_le_bytes(header[16..24].try_into().unwrap()))
            .map_err(|_| invalid_data("id index entry count exceeds usize"))?;
        cursor = header_end;

        let name = interner
            .try_resolve(crate::graph::schema::InternedKey::from_u64(type_key))
            .ok_or_else(|| invalid_data("id index contains an unresolved type key"))?
            .to_string();

        match variant_tag {
            0 => {
                let keys_size = 4usize
                    .checked_mul(num_entries)
                    .ok_or_else(|| invalid_data("id index integer key size overflow"))?;
                let block_size = keys_size
                    .checked_mul(2)
                    .ok_or_else(|| invalid_data("id index integer block size overflow"))?;
                let block_end = cursor
                    .checked_add(block_size)
                    .ok_or_else(|| invalid_data("id index integer block offset overflow"))?;
                if block_end > payload.len() {
                    return Err(invalid_data("id_indices Integer block truncated"));
                }
                let keys_bytes = &payload[cursor..cursor + keys_size];
                let idxs_bytes = &payload[cursor + keys_size..block_end];
                cursor = block_end;
                let mut map =
                    FxHashMap::<u32, petgraph::graph::NodeIndex>::with_capacity_and_hasher(
                        num_entries,
                        Default::default(),
                    );
                let mut previous = None;
                for i in 0..num_entries {
                    let k = u32::from_le_bytes(keys_bytes[i * 4..i * 4 + 4].try_into().unwrap());
                    if previous.is_some_and(|prior| k <= prior) {
                        return Err(invalid_data(
                            "id index integer keys are not strictly increasing",
                        ));
                    }
                    previous = Some(k);
                    let v = u32::from_le_bytes(idxs_bytes[i * 4..i * 4 + 4].try_into().unwrap())
                        as usize;
                    map.insert(k, petgraph::graph::NodeIndex::new(v));
                }
                if out.insert(name, TypeIdIndex::Integer(map)).is_some() {
                    return Err(invalid_data("id index contains duplicate type names"));
                }
            }
            1 => {
                let length_end = cursor
                    .checked_add(8)
                    .ok_or_else(|| invalid_data("general blob length offset overflow"))?;
                let length_bytes = payload
                    .get(cursor..length_end)
                    .ok_or_else(|| invalid_data("id_indices General block missing blob length"))?;
                let blob_len =
                    usize::try_from(u64::from_le_bytes(length_bytes.try_into().unwrap()))
                        .map_err(|_| invalid_data("general blob length exceeds usize"))?;
                if blob_len as u64 > 2 * 1024 * 1024 * 1024 {
                    return Err(invalid_data("general id index exceeds decode limit"));
                }
                cursor = length_end;
                let blob_end = cursor
                    .checked_add(blob_len)
                    .ok_or_else(|| invalid_data("general blob range overflow"))?;
                let blob = payload
                    .get(cursor..blob_end)
                    .ok_or_else(|| invalid_data("id_indices General blob truncated"))?;
                let _ = (name, blob, num_entries);
                // This pre-0.14 cache encoded general IDs with bincode. The
                // caller treats `None` as a cache miss and rebuilds lazily.
                return Ok(None);
            }
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("id_indices unknown variant tag {}", other),
                ));
            }
        }
    }
    if cursor != payload.len() {
        return Err(invalid_data("id_indices.bin.zst has trailing bytes"));
    }
    Ok(Some(out))
}

// ─── type_connectivity.bin.zst (0.8.13 fast-load) ────────────────────────────
//
// Replaces a 266 MB JSON array of `ConnectivityTriple { src: String, conn:
// String, tgt: String, count: usize }` embedded inside metadata.json with
// a compact binary file at the graph root. The old metadata.json path
// still loads on fallback so graphs saved by 0.8.11 / 0.8.12 continue to
// open without a rebuild.
//
// Payload (pre-zstd):
//   [ 0..  8]  magic   = b"KGLTCN1\0"
//   [ 8.. 12]  version = u32 LE (= 1)
//   [12.. 16]  n       = u32 LE
//   [16.. n*32+16]  entries: (u64 src_key, u64 conn_key, u64 tgt_key, u64 count) * n
//
// `src_key`/`conn_key`/`tgt_key` are interner hashes produced by
// `InternedKey::as_u64()`; the load path resolves them via
// `graph.interner.try_resolve`. The interner is always loaded before
// this file on the disk-load path (`load_disk_dir`).

const TYPE_CONN_MAGIC: &[u8; 8] = b"KGLTCN1\0";
const TYPE_CONN_VERSION: u32 = 1;

// secondary_labels.bin.zst format. Persists DirGraph.secondary_label_index
// for disk-backed graphs. Memory + mapped backends carry secondaries inline
// on NodeData in the portable payload; disk's columnar layout has no
// per-row label slot, so we need this sidecar.
//
// Payload layout (zstd-compressed):
//   [0..8]   magic = b"KGLSLBL1"
//   [8..12]  version = 1u32 LE
//   [12..16] num_labels (u32 LE)
//   For each label:
//     [..8]  label_key (u64 LE, raw InternedKey)
//     [..4]  num_nodes (u32 LE)
//     [..]   num_nodes × NodeIndex (u32 LE each)
//
// Resolution: `label_key` is `InternedKey::as_u64()`; the load path
// resolves it via `graph.interner.try_resolve`. Missing interner
// entries are silently skipped (covers truly-corrupted input).
const SECONDARY_LABELS_MAGIC: &[u8; 8] = b"KGLSLBL1";
const SECONDARY_LABELS_VERSION: u32 = 1;

/// Writer for `type_connectivity.bin.zst`. Idempotent — no-op if the
/// cache is empty. Called from `DirGraph::save_disk` after
/// `metadata.json` is emitted.
pub(crate) fn write_type_connectivity_bin(
    dir: &std::path::Path,
    graph: &DirGraph,
) -> Result<(), String> {
    let Some(triples) = graph.get_type_connectivity() else {
        return Ok(());
    };
    if triples.is_empty() {
        return Ok(());
    }
    let n = triples.len() as u32;
    let mut payload: Vec<u8> = Vec::with_capacity(16 + (triples.len() * 32));
    payload.extend_from_slice(TYPE_CONN_MAGIC);
    payload.extend_from_slice(&TYPE_CONN_VERSION.to_le_bytes());
    payload.extend_from_slice(&n.to_le_bytes());
    // Intern each string once; avoids 3*N lookups if the interner's
    // `get_or_intern` hashes the string internally.
    let mut interner = graph.interner.clone();
    for t in &triples {
        let src_key = interner
            .try_get_or_intern(&t.src)
            .map_err(|e| e.to_string())?
            .as_u64();
        let conn_key = interner
            .try_get_or_intern(&t.conn)
            .map_err(|e| e.to_string())?
            .as_u64();
        let tgt_key = interner
            .try_get_or_intern(&t.tgt)
            .map_err(|e| e.to_string())?
            .as_u64();
        payload.extend_from_slice(&src_key.to_le_bytes());
        payload.extend_from_slice(&conn_key.to_le_bytes());
        payload.extend_from_slice(&tgt_key.to_le_bytes());
        payload.extend_from_slice(&(t.count as u64).to_le_bytes());
    }
    let compressed = zstd::encode_all(payload.as_slice(), 3)
        .map_err(|e| format!("type_connectivity compression failed: {}", e))?;
    std::fs::write(dir.join("type_connectivity.bin.zst"), compressed)
        .map_err(|e| format!("Failed to write type_connectivity.bin.zst: {}", e))?;
    Ok(())
}

/// Reader for `type_connectivity.bin.zst`. Returns `Ok(None)` if the
/// file is absent or has an unrecognised magic tag (caller falls back
/// to the metadata JSON representation).
pub(crate) fn read_type_connectivity_bin(
    dir: &std::path::Path,
    graph: &DirGraph,
) -> io::Result<Option<Vec<crate::graph::schema::ConnectivityTriple>>> {
    let path = dir.join("type_connectivity.bin.zst");
    if !path.exists() {
        return Ok(None);
    }
    let compressed = std::fs::read(&path)?;
    let payload = zstd_decompress(&compressed)?;
    if payload.len() < 16 || &payload[..8] != TYPE_CONN_MAGIC {
        return Ok(None);
    }
    let version = u32::from_le_bytes(payload[8..12].try_into().unwrap());
    if version != TYPE_CONN_VERSION {
        return Ok(None);
    }
    let n = u32::from_le_bytes(payload[12..16].try_into().unwrap()) as usize;
    let entry_bytes = 32usize;
    let expected_len = 16 + n * entry_bytes;
    if payload.len() < expected_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "type_connectivity.bin.zst is truncated",
        ));
    }
    let mut triples = Vec::with_capacity(n);
    for i in 0..n {
        let base = 16 + i * entry_bytes;
        let src_key = u64::from_le_bytes(payload[base..base + 8].try_into().unwrap());
        let conn_key = u64::from_le_bytes(payload[base + 8..base + 16].try_into().unwrap());
        let tgt_key = u64::from_le_bytes(payload[base + 16..base + 24].try_into().unwrap());
        let count = u64::from_le_bytes(payload[base + 24..base + 32].try_into().unwrap());
        let src = graph
            .interner
            .try_resolve(crate::graph::schema::InternedKey::from_u64(src_key))
            .map(|s| s.to_string());
        let conn = graph
            .interner
            .try_resolve(crate::graph::schema::InternedKey::from_u64(conn_key))
            .map(|s| s.to_string());
        let tgt = graph
            .interner
            .try_resolve(crate::graph::schema::InternedKey::from_u64(tgt_key))
            .map(|s| s.to_string());
        if let (Some(src), Some(conn), Some(tgt)) = (src, conn, tgt) {
            triples.push(crate::graph::schema::ConnectivityTriple {
                src,
                conn,
                tgt,
                count: count as usize,
            });
        }
        // Missing interner entry → silently skip. The interner is loaded
        // before this file, so this only trips on truly corrupted input.
    }
    Ok(Some(triples))
}

/// Encode `DirGraph.secondary_label_index` into a self-describing
/// byte payload. Returns `None` if the graph has no secondary
/// labels — callers skip writing the section entirely, keeping
/// single-label graphs zero-cost.
///
/// Labels are stored as length-prefixed UTF-8 strings (not raw
/// InternedKey u64s) because secondary-only labels aren't carried
/// by any other persisted structure — the load-side interner
/// wouldn't recognise the key otherwise. Strings are intern-cheap
/// (one string per label, not per node).
///
/// Layout (uncompressed):
///   [0..8]    magic (`b"KGLSLBL1"`)
///   [8..12]   version (`1u32` LE)
///   [12..16]  num_labels (u32 LE)
///   For each label:
///     4 B   name_len (u32 LE)
///     name_len B   UTF-8 label name
///     4 B   num_nodes (u32 LE)
///     4*N B node indices (raw `NodeIndex::index() as u32` LE)
///
/// Used by both the disk sidecar (`secondary_labels.bin.zst`) and
/// the in-memory `.kgl` v4 envelope's secondary-labels section.
pub(super) fn encode_secondary_label_index(graph: &DirGraph) -> Option<Vec<u8>> {
    if !graph.has_secondary_labels || graph.secondary_label_index.is_empty() {
        return None;
    }
    let n = graph.secondary_label_index.len() as u32;
    let mut payload: Vec<u8> = Vec::new();
    payload.extend_from_slice(SECONDARY_LABELS_MAGIC);
    payload.extend_from_slice(&SECONDARY_LABELS_VERSION.to_le_bytes());
    payload.extend_from_slice(&n.to_le_bytes());
    // Deterministic order: sort by label name (string) so byte
    // layout is stable across saves of the same logical state.
    let mut entries: Vec<(
        &crate::graph::schema::InternedKey,
        &Vec<petgraph::graph::NodeIndex>,
    )> = graph.secondary_label_index.iter().collect();
    entries.sort_by_key(|(k, _)| graph.interner.resolve(**k).to_string());
    for (key, nodes) in entries {
        let name = graph.interner.resolve(*key);
        let name_bytes = name.as_bytes();
        payload.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        payload.extend_from_slice(name_bytes);
        payload.extend_from_slice(&(nodes.len() as u32).to_le_bytes());
        for idx in nodes {
            payload.extend_from_slice(&(idx.index() as u32).to_le_bytes());
        }
    }
    Some(payload)
}

/// Decode a `secondary_label_index` payload into the graph in
/// place. Interns each label name through the graph's live
/// interner — so even labels that exist *only* as secondaries
/// (no node has them as primary type) round-trip correctly.
/// Returns `Ok(false)` if the header doesn't match (graceful —
/// older saves don't have the section).
pub(super) fn decode_secondary_label_index(
    payload: &[u8],
    graph: &mut DirGraph,
) -> io::Result<bool> {
    if payload.len() < 16 || &payload[..8] != SECONDARY_LABELS_MAGIC {
        return Ok(false);
    }
    let version = u32::from_le_bytes(payload[8..12].try_into().unwrap());
    if version != SECONDARY_LABELS_VERSION {
        return Ok(false);
    }
    let n = u32::from_le_bytes(payload[12..16].try_into().unwrap()) as usize;
    let mut cursor = 16usize;
    let mut index: HashMap<crate::graph::schema::InternedKey, Vec<petgraph::graph::NodeIndex>> =
        HashMap::with_capacity(n);
    for _ in 0..n {
        if payload.len() < cursor + 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "secondary_labels payload truncated (name len)",
            ));
        }
        let name_len = u32::from_le_bytes(payload[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;
        if payload.len() < cursor + name_len + 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "secondary_labels payload truncated (name bytes)",
            ));
        }
        let name = std::str::from_utf8(&payload[cursor..cursor + name_len])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
            .to_string();
        cursor += name_len;
        let num_nodes =
            u32::from_le_bytes(payload[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;
        if payload.len() < cursor + num_nodes * 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "secondary_labels payload truncated (node list)",
            ));
        }
        let key = graph
            .interner
            .try_get_or_intern(&name)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let mut nodes = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            let raw = u32::from_le_bytes(payload[cursor..cursor + 4].try_into().unwrap());
            cursor += 4;
            nodes.push(petgraph::graph::NodeIndex::new(raw as usize));
        }
        index.insert(key, nodes);
    }
    // Heal dangling indices. A graph saved by a version that deleted a
    // labelled node without evicting it from this index (pre-0.10.6) carries
    // stale NodeIndex entries pointing at now-absent nodes. NodeData does not
    // carry the labels (this index is canonical), so we can't rebuild — but
    // we can drop indices whose node is gone, mirroring the live-node retain
    // pattern used elsewhere. Nodes are fully loaded before this runs.
    {
        // Arena guard: node_weight materializes on the disk backend
        // (protocol in disk/graph.rs); no-op on the memory/mapped graphs
        // this load path produces. Scoped so the borrow ends before the
        // &mut assignments below.
        let _arena_guard = graph.graph.begin_query();
        for bucket in index.values_mut() {
            bucket.retain(|idx| graph.graph.node_weight(*idx).is_some());
        }
    }
    index.retain(|_, bucket| !bucket.is_empty());
    if !index.is_empty() {
        graph.secondary_label_index = index;
        graph.has_secondary_labels = true;
    }
    Ok(true)
}

/// Disk-mode writer for `secondary_labels.bin.zst`. No-op if the
/// graph has no secondary labels.
pub(crate) fn write_secondary_labels_bin(
    dir: &std::path::Path,
    graph: &DirGraph,
) -> Result<(), String> {
    let Some(payload) = encode_secondary_label_index(graph) else {
        return Ok(());
    };
    let compressed = zstd::encode_all(payload.as_slice(), 3)
        .map_err(|e| format!("secondary_labels compression failed: {}", e))?;
    std::fs::write(dir.join("secondary_labels.bin.zst"), compressed)
        .map_err(|e| format!("Failed to write secondary_labels.bin.zst: {}", e))?;
    Ok(())
}

/// Disk-mode reader for `secondary_labels.bin.zst`. Returns
/// `Ok(false)` if the file is absent (graceful — older disk graphs
/// don't have it). A file that exists but doesn't decode — bad zstd,
/// truncated payload, or wrong magic/version — is corruption and
/// errors (the sidecar is written whole with its magic by every
/// version that emits it, so "present but unrecognisable" is never a
/// legitimate state).
pub(crate) fn read_secondary_labels_bin(
    dir: &std::path::Path,
    graph: &mut DirGraph,
) -> io::Result<bool> {
    let path = dir.join("secondary_labels.bin.zst");
    if !path.exists() {
        return Ok(false);
    }
    let compressed = std::fs::read(&path)?;
    let payload = zstd_decompress(&compressed)?;
    match decode_secondary_label_index(&payload, graph)? {
        true => Ok(true),
        false => Err(invalid_data(
            "secondary_labels.bin.zst decompressed but its header is unrecognised",
        )),
    }
}

#[cfg(test)]
mod interner_file_tests {
    use super::*;

    #[test]
    fn malformed_interner_collision_is_invalid_data() {
        let dir = tempfile::tempdir().unwrap();
        let incoming = "persisted-name";
        let bytes = encode_disk_serde(&vec![incoming.to_string()]).unwrap();
        let compressed = zstd::encode_all(bytes.as_slice(), 3).unwrap();
        std::fs::write(dir.path().join("interner.bin.zst"), compressed).unwrap();

        let mut graph = DirGraph::new();
        graph
            .interner
            .try_register(
                crate::graph::schema::InternedKey::from_str(incoming),
                "conflicting-existing",
            )
            .unwrap();
        let err = read_interner_bin(dir.path(), &mut graph).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("hash collision"));
    }

    #[test]
    fn disk_sidecar_frame_is_versioned_and_unframed_payloads_are_rejected() {
        let values = vec!["alpha".to_string(), "beta".to_string()];
        let current = encode_disk_serde(&values).unwrap();
        assert_eq!(&current[..8], DISK_SERDE_MAGIC);
        assert_eq!(current[8], serde_codec::CodecVersion::PostcardV1.tag());
        assert_eq!(
            decode_disk_serde::<Vec<String>>(&current, current.capacity() as u64).unwrap(),
            values
        );

        let unframed =
            serde_codec::encode_versioned(serde_codec::CURRENT_CODEC, &values, MAX_CODEC_BYTES)
                .unwrap();
        let error =
            decode_disk_serde::<Vec<String>>(&unframed, unframed.capacity() as u64).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("pre-0.14"));
    }

    #[test]
    fn disk_sidecar_frame_rejects_unknown_codec_and_trailing_bytes() {
        let mut unknown = encode_disk_serde(&vec![1u32, 2]).unwrap();
        unknown[8] = 99;
        let error = decode_disk_serde::<Vec<u32>>(&unknown, unknown.capacity() as u64).unwrap_err();
        assert!(error
            .to_string()
            .contains("unknown binary codec version 99"));

        let mut trailing = encode_disk_serde(&vec![1u32, 2]).unwrap();
        trailing.push(0xff);
        let error =
            decode_disk_serde::<Vec<u32>>(&trailing, trailing.capacity() as u64).unwrap_err();
        assert!(error.to_string().contains("trailing"));
    }
}
