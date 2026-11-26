//! Vector-cache and standalone embedding-file persistence.

use super::{codec_deser, codec_ser, MAX_CODEC_BYTES};
use crate::datatypes::values::Value;
use crate::graph::algorithms::hnsw::HnswIndex;
use crate::graph::edge_embeddings::carry::{
    extract_edge_stores, install_edge_stores, resolve_edge_stores, CarriedEdgeStore,
    RelationshipCarryStats, RelationshipKeys,
};
use crate::graph::embedding_validation::validate_finite_vector;
use crate::graph::index_freshness::IndexFreshness;
use crate::graph::schema::{DirGraph, EmbeddingStore};
use crate::graph::storage::GraphRead;
use crate::serde_codec;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

// ─── HNSW vector-index section (0.11.0) ───────────────────────────────────
//
// A self-describing, *skippable* `.kgl` sub-section carrying built HNSW
// indexes. The whole point is robustness against future change: the index is a
// rebuildable cache, never a correctness dependency, so any version mismatch or
// corruption is silently dropped (the store loads fine without an index; the
// user rebuilds, or auto-use just doesn't fire). Bumping
// `VECTOR_INDEX_FORMAT_VERSION` lets the on-disk index format evolve WITHOUT a
// core-data-version bump — older readers skip a newer index, newer readers skip
// an older one.
//
//   [0..8]   magic = b"KGLVIDX1"
//   [8..12]  format_version: u32 LE
//   [12..]   codec payload for Vec<PersistedVectorIndex>
//
// v3 (0.16.10) adds the catch-up state beside the topology. An index no longer
// has to cover its whole store — it covers a prefix and remembers the rest —
// so a file that carried only the topology can no longer be interpreted: a v2
// payload would restore an index while its outstanding delta was lost, and the
// vectors written after the save would look indexed. v2 files therefore drop
// their index and rebuild, which is exactly the rebuildable-cache contract this
// section was designed around.
pub(super) const VECTOR_INDEX_MAGIC: &[u8; 8] = b"KGLVIDX1";
const VECTOR_INDEX_FORMAT_VERSION: u32 = 3;

/// One store's index held open for the duration of an encode. The read guard
/// is what keeps a concurrent catch-up from renumbering topology mid-write.
struct HeldIndex<'a> {
    node_type: &'a str,
    embedding_property: &'a str,
    guard: crate::graph::schema::HnswRead<'a>,
    watermark: u32,
    limit: usize,
    dirty: Vec<u32>,
}

/// One store's persisted index, as written — borrowed so a save does not clone
/// a corpus-sized topology. Postcard encodes struct fields positionally, so
/// this and [`PersistedVectorIndex`] are the same bytes.
#[derive(Serialize)]
struct PersistedVectorIndexRef<'a> {
    node_type: &'a str,
    embedding_property: &'a str,
    index: &'a HnswIndex,
    watermark: u32,
    limit: usize,
    dirty: Vec<u32>,
}

/// One store's persisted index: the topology plus what it has yet to cover.
/// The same payload carries relationship indexes in their own section
/// (`edge_vector_persistence`), where `node_type` holds the relationship type.
#[derive(Serialize, Deserialize)]
pub(super) struct PersistedVectorIndex {
    node_type: String,
    embedding_property: String,
    index: HnswIndex,
    /// Store slots the index covers.
    watermark: u32,
    /// The inline-refresh ceiling this index was built with.
    limit: usize,
    /// Slots replaced in place since the last catch-up. Sorted on write so
    /// equivalent graphs serialize byte-identically.
    dirty: Vec<u32>,
}

impl PersistedVectorIndex {
    /// The `(type, embedding property)` store key this index was saved under.
    pub(super) fn take_key(&mut self) -> (String, String) {
        (
            std::mem::take(&mut self.node_type),
            std::mem::take(&mut self.embedding_property),
        )
    }
}

/// Encode every built HNSW index into a self-describing payload. Returns `None`
/// when no store carries an index (the section is then omitted entirely).
pub(super) fn encode_vector_indexes(graph: &DirGraph) -> io::Result<Option<Vec<u8>>> {
    encode_index_payload(
        graph
            .embeddings
            .iter()
            .map(|((nt, prop), store)| (nt.as_str(), prop.as_str(), store)),
    )
}

/// Encode the built indexes among `stores`, keyed by `(type, property)`, into
/// a KGLVIDX1 payload; `None` when none is built.
pub(super) fn encode_index_payload<'a>(
    stores: impl Iterator<Item = (&'a str, &'a str, &'a EmbeddingStore)>,
) -> io::Result<Option<Vec<u8>>> {
    // Key-sorted so a multi-store graph serializes byte-identically; the
    // underlying map's iteration order is per-process.
    let mut stores: Vec<_> = stores.collect();
    stores.sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    // The read guards are held for the encode: they are what keeps a
    // concurrent catch-up from renumbering topology mid-serialization.
    let held: Vec<HeldIndex<'_>> = stores
        .into_iter()
        .filter_map(|(nt, prop, store)| {
            let (watermark, limit, dirty) = store.freshness_state().persisted_parts();
            Some(HeldIndex {
                node_type: nt,
                embedding_property: prop,
                guard: store.index_read()?,
                watermark,
                limit,
                dirty,
            })
        })
        .collect();
    if held.is_empty() {
        return Ok(None);
    }
    let entries: Vec<PersistedVectorIndexRef<'_>> = held
        .iter()
        .map(|held| PersistedVectorIndexRef {
            node_type: held.node_type,
            embedding_property: held.embedding_property,
            index: &held.guard,
            watermark: held.watermark,
            limit: held.limit,
            dirty: held.dirty.clone(),
        })
        .collect();
    let body = codec_ser(serde_codec::CodecVersion::PostcardV1, &entries)?;
    let mut payload = Vec::with_capacity(12 + body.len());
    payload.extend_from_slice(VECTOR_INDEX_MAGIC);
    payload.extend_from_slice(&VECTOR_INDEX_FORMAT_VERSION.to_le_bytes());
    payload.extend_from_slice(&body);
    Ok(Some(payload))
}

/// Decode the vector-index section and attach indexes to the matching stores.
/// Best-effort: an unrecognised magic, an unknown format version, a codec
/// error, or a shape mismatch against the loaded store all result in the index
/// being silently skipped — never a load failure. Must run AFTER embeddings are
/// loaded and their norms rebuilt (cosine navigation needs the norm cache).
pub(super) fn decode_vector_indexes(payload: &[u8], graph: &mut DirGraph) {
    for mut entry in decode_index_payload(payload) {
        if let Some(store) = graph.embeddings.get_mut(&entry.take_key()) {
            attach_decoded_index(store, entry);
        }
    }
}

/// Decode a KGLVIDX1 payload. An unrecognised magic, an unknown format version
/// or a codec error yields no entries: the index is a rebuildable cache.
pub(super) fn decode_index_payload(payload: &[u8]) -> Vec<PersistedVectorIndex> {
    if payload.len() < 12 || &payload[..8] != VECTOR_INDEX_MAGIC {
        return Vec::new();
    }
    let ver = u32::from_le_bytes([payload[8], payload[9], payload[10], payload[11]]);
    if ver != VECTOR_INDEX_FORMAT_VERSION {
        return Vec::new(); // rebuildable cache: skip unknown and pre-0.16.10 versions
    }
    let codec = serde_codec::CodecVersion::PostcardV1;
    codec_deser(codec, &payload[12..], (payload.len() - 12) as u64).unwrap_or_default()
}

/// Attach one decoded index to the store it was saved for, or drop it.
pub(super) fn attach_decoded_index(store: &mut EmbeddingStore, entry: PersistedVectorIndex) {
    // Defensive: only attach an index whose shape still matches the store it
    // was built over (dimension + a coverage that the store's vectors actually
    // contain), and whose recorded coverage agrees with the topology's own
    // length — a watermark ahead of the index would silence a delta that was
    // never folded in.
    let shape_ok = entry
        .index
        .validate_for_store(&store.data, &store.norms, store.dimension)
        .is_ok();
    if !shape_ok
        || entry.watermark as usize != entry.index.len()
        || entry.dirty.iter().any(|slot| *slot >= entry.watermark)
    {
        return;
    }
    let freshness = IndexFreshness::restored(entry.watermark, entry.limit, &entry.dirty);
    store.attach_persisted_index(entry.index, freshness);
}

// ─── Embedding Export / Import ────────────────────────────────────────────

/// Magic bytes for the embedding export format.
const KGLE_MAGIC: [u8; 4] = *b"KGLE";
/// v3 selects Postcard and includes store/vector provenance. A node-only export
/// still writes exactly v3, so older readers keep reading it.
const KGLE_NODE_ONLY_VERSION: u32 = 3;
/// v4 adds relationship stores ([`KgleV4Payload`]). Written only when the
/// export carries one; 0.17.12 and older refuse it by version number
/// ("newer than supported version 3") instead of misreading it.
const KGLE_VERSION: u32 = 4;

/// The v4 payload root: node stores as in v3, then relationship stores
/// addressed by endpoint ids (see `edge_embeddings::carry`).
#[derive(Serialize, Deserialize)]
struct KgleV4Payload {
    nodes: Vec<ExportedEmbeddingStore>,
    relationships: Vec<CarriedEdgeStore>,
}

/// A single embedding store serialized with node IDs (not internal indices).
/// v2 adds provenance: the store `metric`/`model_id` and a per-entry text hash,
/// so `import_embeddings` round-trips what `embed_texts(mode='changed')` needs.
#[derive(Serialize, Deserialize)]
struct ExportedEmbeddingStore {
    node_type: String,
    text_column: String, // e.g. "summary" (without _emb suffix)
    dimension: usize,
    /// Store default metric (`set_embeddings(metric=…)`), `None` if unset.
    metric: Option<String>,
    /// Embedder id stamped by `embed_texts`, `None` for raw-vector stores.
    model_id: Option<String>,
    /// (node_id, embedding, optional source-text hash). The hash is `Some` only
    /// for vectors produced by `embed_texts` (drives `mode='changed'`).
    entries: Vec<(Value, Vec<f32>, Option<u64>)>,
}

/// Filter for selective embedding export.
pub enum EmbeddingExportFilter {
    /// Export all embedding stores for these node types.
    Types(Vec<String>),
    /// Export specific (node_type → [text_columns]) pairs.
    /// An empty vec means all properties for that type.
    TypeProperties(HashMap<String, Vec<String>>),
}

pub struct ExportStats {
    pub stores: usize,
    pub embeddings: usize,
    /// Relationship stores written (the file is v4 when this is non-zero).
    pub relationship_stores: usize,
    pub relationship_embeddings: usize,
}

pub struct ImportStats {
    pub stores: usize,
    pub imported: usize,
    pub skipped: usize,
    /// Number of stores in the file whose entries all failed to match
    /// nodes in the current graph (so the store was dropped and not
    /// inserted into `graph.embeddings`). Surfaces the silent-drop
    /// case where the .kgle file was exported from a graph with
    /// different node IDs or types — the count of such stores would
    /// otherwise be invisible to callers.
    pub dropped_stores: usize,
    /// `(node type, text column)` of each node store installed.
    pub installed: Vec<(String, String)>,
    /// The relationship-store carry (all zero for a v3 file).
    pub relationships: RelationshipCarryStats,
}

/// The decoded payload of any readable version: v3 carries node stores only.
struct DecodedEmbeddingFile {
    nodes: Vec<ExportedEmbeddingStore>,
    relationships: Vec<CarriedEdgeStore>,
}

fn decode_embedding_file_payload(buf: &[u8], version: u32) -> io::Result<DecodedEmbeddingFile> {
    match version {
        KGLE_NODE_ONLY_VERSION => Ok(DecodedEmbeddingFile {
            nodes: decode_embedding_file_body(buf, version)?,
            relationships: Vec::new(),
        }),
        KGLE_VERSION => {
            let root: KgleV4Payload = decode_embedding_file_body(buf, version)?;
            Ok(DecodedEmbeddingFile {
                nodes: root.nodes,
                relationships: root.relationships,
            })
        }
        newer if newer > KGLE_VERSION => Err(io::Error::other(format!(
            "Embedding file version {newer} is newer than supported version {KGLE_VERSION}. \
             Please upgrade kglite."
        ))),
        older => Err(super::pre_014_bincode_error(
            format!(".kgle embedding file v{older}").as_str(),
        )),
    }
}

fn decode_embedding_file_body<T: serde::de::DeserializeOwned>(
    buf: &[u8],
    version: u32,
) -> io::Result<T> {
    if buf.len() < 9 {
        return Err(io::Error::other(format!(
            "Embedding file v{version} is truncated before its codec tag."
        )));
    }
    let codec = serde_codec::CodecVersion::from_tag(buf[8])
        .map_err(|e| io::Error::other(format!("Invalid .kgle codec tag: {e}")))?;
    let decoder = GzDecoder::new(&buf[9..]);
    let mut bounded = decoder.take(MAX_CODEC_BYTES.saturating_add(1));
    let mut payload = Vec::new();
    bounded.read_to_end(&mut payload)?;
    if payload.len() as u64 > MAX_CODEC_BYTES {
        return Err(io::Error::other(format!(
            "Decompressed embedding payload exceeds the {MAX_CODEC_BYTES} byte limit"
        )));
    }
    codec_deser(codec, &payload, payload.capacity() as u64)
        .map_err(|e| io::Error::other(format!("Failed to deserialize embedding data: {e}")))
}

/// An ambiguous relationship is the caller's to resolve (name a key), so it is
/// `InvalidInput` rather than a file error; bindings surface it as an argument
/// error.
fn carry_refusal(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

fn validate_exported_embedding_stores(stores: &[ExportedEmbeddingStore]) -> io::Result<()> {
    for store in stores {
        for (entry, (_, vector, _)) in store.entries.iter().enumerate() {
            if vector.len() != store.dimension {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "Invalid embedding in store '{}.{}' at entry {entry}: expected width {}, got {}",
                        store.node_type,
                        store.text_column,
                        store.dimension,
                        vector.len()
                    ),
                ));
            }
            validate_finite_vector(vector).map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "Invalid embedding in store '{}.{}' at entry {entry}: {error}",
                        store.node_type, store.text_column
                    ),
                )
            })?;
        }
    }
    Ok(())
}

/// Export embeddings to a standalone .kgle file, keyed by node ID.
pub fn export_embeddings_to_file(
    graph: &DirGraph,
    path: &str,
    filter: Option<&EmbeddingExportFilter>,
    relationship_keys: &RelationshipKeys,
) -> io::Result<ExportStats> {
    // Relationship stores first: a parallel group without a usable key refuses
    // the whole export before a byte is written. A node-type filter selects
    // node stores only, so it exports no relationship store.
    let relationships = match filter {
        Some(_) => Vec::new(),
        None => extract_edge_stores(graph, relationship_keys).map_err(carry_refusal)?,
    };
    let mut exported_stores: Vec<ExportedEmbeddingStore> = Vec::new();
    let mut total_embeddings = 0usize;

    // Iterate stores in key order: `graph.embeddings` is a HashMap, and an
    // unsorted walk would randomize the exported store sequence per process,
    // breaking `.kgle` byte-reproducibility for multi-store graphs.
    let mut stores_sorted: Vec<_> = graph.embeddings.iter().collect();
    stores_sorted.sort_unstable_by(|a, b| a.0.cmp(b.0));
    for ((node_type, store_name), store) in stores_sorted {
        let text_column =
            crate::graph::embeddings::text_column_of(store_name).unwrap_or(store_name.as_str());

        // Apply filter
        if let Some(f) = filter {
            match f {
                EmbeddingExportFilter::Types(types) => {
                    if !types.iter().any(|t| t == node_type) {
                        continue;
                    }
                }
                EmbeddingExportFilter::TypeProperties(map) => {
                    match map.get(node_type) {
                        None => continue, // type not in filter
                        Some(props) if !props.is_empty() => {
                            if !props.iter().any(|p| p == text_column) {
                                continue;
                            }
                        }
                        Some(_) => {} // empty list = all properties for this type
                    }
                }
            }
        }

        // Resolve node indices → node IDs, carrying each node's text hash.
        let mut entries: Vec<(Value, Vec<f32>, Option<u64>)> = Vec::with_capacity(store.len());
        for &node_index in &store.slot_to_node {
            if let Some(node) = graph
                .graph
                .node_view(petgraph::graph::NodeIndex::new(node_index))
            {
                if let Some(embedding) = store.get_embedding(node_index) {
                    let hash = store.text_hashes.get(&node_index).copied();
                    entries.push((node.id().into_owned(), embedding.to_vec(), hash));
                }
            }
        }

        total_embeddings += entries.len();
        exported_stores.push(ExportedEmbeddingStore {
            node_type: node_type.clone(),
            text_column: text_column.to_string(),
            dimension: store.dimension,
            metric: store.metric.clone(),
            model_id: store.model_id.clone(),
            entries,
        });
    }

    let relationship_stores = relationships.len();
    let relationship_embeddings = relationships.iter().map(|s| s.entries.len()).sum();
    let node_stores = exported_stores.len();
    let (version, payload) = if relationships.is_empty() {
        let payload = codec_ser(serde_codec::CodecVersion::PostcardV1, &exported_stores);
        (KGLE_NODE_ONLY_VERSION, payload)
    } else {
        let root = KgleV4Payload {
            nodes: exported_stores,
            relationships,
        };
        (
            KGLE_VERSION,
            codec_ser(serde_codec::CodecVersion::PostcardV1, &root),
        )
    };
    let payload =
        payload.map_err(|e| io::Error::other(format!("Failed to serialize embeddings: {e}")))?;

    // Write: magic + version + codec tag + gzip(codec(payload)), to a sibling
    // temp that is fsynced and renamed over `path`, so a crash or full disk
    // never leaves a torn `.kgle` where a good one stood.
    write_file_atomically(Path::new(path), |writer| {
        writer.write_all(&KGLE_MAGIC)?;
        writer.write_all(&version.to_le_bytes())?;
        writer.write_all(&[serde_codec::CodecVersion::PostcardV1.tag()])?;
        let mut gz = GzEncoder::new(&mut *writer, Compression::new(3));
        gz.write_all(&payload)?;
        gz.finish()?;
        Ok(())
    })?;

    Ok(ExportStats {
        stores: node_stores,
        embeddings: total_embeddings,
        relationship_stores,
        relationship_embeddings,
    })
}

/// Write `dest` through a same-directory temp (named like a save temp, so
/// `reap_stale_save_temps` also collects a crashed export's), fsync it and
/// rename it into place; the temp is removed on any failure.
fn write_file_atomically(
    dest: &Path,
    write: impl FnOnce(&mut BufWriter<File>) -> io::Result<()>,
) -> io::Result<()> {
    static EXPORT_COUNTER: AtomicU64 = AtomicU64::new(0);
    let nonce = EXPORT_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp_name = format!(
        "{}{}.{}",
        super::save_temp_prefix(dest),
        std::process::id(),
        nonce
    );
    let dir = dest.parent().filter(|p| !p.as_os_str().is_empty());
    let tmp = match dir {
        Some(d) => d.join(&tmp_name),
        None => PathBuf::from(&tmp_name),
    };
    let result = (|| -> io::Result<()> {
        let mut writer = BufWriter::new(File::create(&tmp)?);
        write(&mut writer)?;
        writer.flush()?;
        let file = writer
            .into_inner()
            .map_err(|e| io::Error::other(e.to_string()))?;
        file.sync_all()?;
        std::fs::rename(&tmp, dest)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
        return result;
    }
    crate::graph::durable_io::sync_dir(dir.unwrap_or_else(|| Path::new(".")))
}

/// Import embeddings from a .kgle file, resolving node IDs to current graph indices.
pub fn import_embeddings_from_file(
    graph: &mut DirGraph,
    path: &str,
    relationship_keys: &RelationshipKeys,
) -> io::Result<ImportStats> {
    let file = File::open(path)?;
    let file_len = file.metadata().map(|m| m.len() as usize).unwrap_or(0);
    let mut reader = BufReader::new(file);
    let mut buf = Vec::with_capacity(file_len);
    reader.read_to_end(&mut buf)?;

    if buf.len() < 8 {
        return Err(io::Error::other(
            "File is too small to be a valid .kgle file.",
        ));
    }

    // Validate magic and version
    if buf[..4] != KGLE_MAGIC {
        return Err(io::Error::other(
            "Not a valid .kgle file (bad magic bytes).",
        ));
    }
    let version = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
    let decoded = decode_embedding_file_payload(&buf, version)?;
    // The compressed file bytes are dead once decoded; at store scale they are
    // hundreds of MB that would otherwise stay live through the install loop.
    drop(buf);
    let exported_stores = decoded.nodes;
    // Preflight the complete payload before touching graph indexes or stores:
    // a malformed later store must not leave earlier stores installed, and an
    // ambiguous relationship refuses the whole import.
    validate_exported_embedding_stores(&exported_stores)?;
    let relationships = resolve_edge_stores(graph, decoded.relationships, relationship_keys)
        .map_err(carry_refusal)?;

    let mut total_imported = 0usize;
    let mut total_skipped = 0usize;
    let mut stores_count = 0usize;
    let mut dropped_stores = 0usize;
    let mut installed = Vec::new();

    for mut exported in exported_stores {
        // Build ID index for this node type so lookup_by_id works
        graph.build_id_index(&exported.node_type);

        let mut store = crate::graph::schema::EmbeddingStore::new(exported.dimension);
        // Restore store-level provenance (v2+; `None` for v1 files).
        store.metric = exported.metric.clone();
        store.model_id = exported.model_id.clone();
        store
            .data
            .reserve(exported.entries.len() * exported.dimension);

        let mut imported = 0usize;
        let mut skipped = 0usize;

        // Consumed one entry at a time so each per-vector buffer is freed as
        // it is copied into the packed store.
        let had_entries = !exported.entries.is_empty();
        for (id, vec, hash) in std::mem::take(&mut exported.entries) {
            match graph.lookup_by_id(&exported.node_type, &id) {
                Some(node_idx) => {
                    store.set_embedding(node_idx.index(), &vec);
                    // Restore the per-node text hash so embed_texts(mode='changed')
                    // can diff against it (the whole point of v2 provenance).
                    if let Some(h) = hash {
                        store.set_text_hash(node_idx.index(), h);
                    }
                    imported += 1;
                }
                None => {
                    skipped += 1;
                }
            }
        }

        if imported > 0 {
            graph.set_embedding_store(&exported.node_type, &exported.text_column, store);
            installed.push((exported.node_type.clone(), exported.text_column.clone()));
            stores_count += 1;
        } else if had_entries {
            dropped_stores += 1;
        }

        total_imported += imported;
        total_skipped += skipped;
    }

    if stores_count > 0 {
        graph.bump_version();
    }
    let relationships = install_edge_stores(graph, relationships).map_err(io::Error::other)?;

    Ok(ImportStats {
        stores: stores_count,
        imported: total_imported,
        skipped: total_skipped,
        dropped_stores,
        installed,
        relationships,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::schema::{EmbeddingStore, NodeData};
    use crate::graph::session::Session;
    use crate::graph::storage::GraphWrite;
    use tempfile::NamedTempFile;

    fn fixture_store() -> ExportedEmbeddingStore {
        ExportedEmbeddingStore {
            node_type: "Doc".to_string(),
            text_column: "summary".to_string(),
            dimension: 2,
            metric: Some("cosine".to_string()),
            model_id: Some("fixture".to_string()),
            entries: vec![(Value::UniqueId(7), vec![0.25, 0.75], Some(99))],
        }
    }

    fn embedding_file(version: u32, codec_tag: Option<u8>, payload: &[u8]) -> Vec<u8> {
        let mut compressed = GzEncoder::new(Vec::new(), Compression::new(3));
        compressed.write_all(payload).unwrap();
        let compressed = compressed.finish().unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&KGLE_MAGIC);
        bytes.extend_from_slice(&version.to_le_bytes());
        if let Some(tag) = codec_tag {
            bytes.push(tag);
        }
        bytes.extend_from_slice(&compressed);
        bytes
    }

    fn graph_with_doc_and_embedding(vector: &[f32]) -> DirGraph {
        let mut graph = DirGraph::new();
        let node = NodeData::new(
            Value::UniqueId(7),
            Value::String("doc".to_string()),
            "Doc".to_string(),
            HashMap::new(),
            &mut graph.interner,
        );
        let idx = GraphWrite::add_node(&mut graph.graph, node);
        graph
            .type_indices
            .entry_or_default("Doc".to_string())
            .push(idx);
        graph.build_id_index("Doc");

        let mut store = EmbeddingStore::new(vector.len());
        store.set_embedding(idx.index(), vector);
        graph
            .embeddings
            .insert(("Doc".to_string(), "summary_emb".to_string()), store);
        graph
    }

    fn write_embedding_file(stores: Vec<ExportedEmbeddingStore>) -> NamedTempFile {
        let payload = codec_ser(serde_codec::CodecVersion::PostcardV1, &stores).unwrap();
        let bytes = embedding_file(
            KGLE_NODE_ONLY_VERSION,
            Some(serde_codec::CodecVersion::PostcardV1.tag()),
            &payload,
        );
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&bytes).unwrap();
        file
    }

    /// Every decoded store must pass shape validation before the first store
    /// is installed. Otherwise an invalid later store makes the operation
    /// fail only after an earlier valid store has already replaced live data.
    #[test]
    fn malformed_vector_width_rejects_the_whole_import_atomically() {
        let stores = vec![
            ExportedEmbeddingStore {
                entries: vec![(Value::UniqueId(7), vec![1.0, 2.0], None)],
                ..fixture_store()
            },
            ExportedEmbeddingStore {
                text_column: "body".to_string(),
                dimension: 2,
                entries: vec![(Value::UniqueId(7), vec![3.0], None)],
                ..fixture_store()
            },
        ];
        let payload = codec_ser(serde_codec::CodecVersion::PostcardV1, &stores).unwrap();
        let bytes = embedding_file(
            KGLE_NODE_ONLY_VERSION,
            Some(serde_codec::CodecVersion::PostcardV1.tag()),
            &payload,
        );
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&bytes).unwrap();

        let mut graph = graph_with_doc_and_embedding(&[9.0, 9.0]);
        let error = match import_embeddings_from_file(
            &mut graph,
            file.path().to_str().unwrap(),
            &RelationshipKeys::new(),
        ) {
            Ok(_) => panic!("a vector whose width differs from its store must be rejected"),
            Err(error) => error,
        };

        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        let idx = graph
            .lookup_by_id_readonly("Doc", &Value::UniqueId(7))
            .unwrap();
        assert_eq!(
            graph.embeddings[&("Doc".to_string(), "summary_emb".to_string())]
                .get_embedding(idx.index()),
            Some(&[9.0, 9.0][..]),
            "a valid store preceding the malformed one must not be installed"
        );
        assert!(
            !graph
                .embeddings
                .contains_key(&("Doc".to_string(), "body_emb".to_string())),
            "the malformed store must not be installed"
        );
    }

    #[test]
    fn non_finite_vector_is_rejected_before_import() {
        let stores = vec![ExportedEmbeddingStore {
            entries: vec![(Value::UniqueId(7), vec![f32::NAN, 2.0], None)],
            ..fixture_store()
        }];
        let payload = codec_ser(serde_codec::CodecVersion::PostcardV1, &stores).unwrap();
        let bytes = embedding_file(
            KGLE_NODE_ONLY_VERSION,
            Some(serde_codec::CodecVersion::PostcardV1.tag()),
            &payload,
        );
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&bytes).unwrap();

        let mut graph = graph_with_doc_and_embedding(&[9.0, 9.0]);
        let error = match import_embeddings_from_file(
            &mut graph,
            file.path().to_str().unwrap(),
            &RelationshipKeys::new(),
        ) {
            Ok(_) => panic!("a non-finite vector must be rejected"),
            Err(error) => error,
        };

        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("must be finite"));
        let idx = graph
            .lookup_by_id_readonly("Doc", &Value::UniqueId(7))
            .unwrap();
        assert_eq!(
            graph.embeddings[&("Doc".to_string(), "summary_emb".to_string())]
                .get_embedding(idx.index()),
            Some(&[9.0, 9.0][..])
        );
    }

    #[test]
    fn embedding_import_publishes_inside_session_transaction() {
        let file = write_embedding_file(vec![fixture_store()]);
        let mut graph = graph_with_doc_and_embedding(&[9.0, 9.0]);
        graph.embeddings.clear();
        let session = Session::new(graph);
        let before = session.version();

        let stats = session
            .transact(|working| {
                import_embeddings_from_file(
                    working,
                    file.path().to_str().unwrap(),
                    &RelationshipKeys::new(),
                )
            })
            .unwrap();

        assert_eq!(stats.stores, 1);
        assert_eq!(stats.imported, 1);
        assert_eq!(session.version(), before + 1);
        let snapshot = session.snapshot();
        let idx = snapshot
            .lookup_by_id_readonly("Doc", &Value::UniqueId(7))
            .unwrap();
        assert_eq!(
            snapshot.embeddings[&("Doc".to_string(), "summary_emb".to_string())]
                .get_embedding(idx.index()),
            Some(&[0.25, 0.75][..])
        );
    }

    #[test]
    fn embedding_import_without_a_matching_vector_is_a_true_no_op() {
        let unmatched = ExportedEmbeddingStore {
            entries: vec![(Value::UniqueId(999), vec![0.25, 0.75], None)],
            ..fixture_store()
        };
        let empty = ExportedEmbeddingStore {
            entries: Vec::new(),
            ..fixture_store()
        };

        for exported in [unmatched, empty] {
            let file = write_embedding_file(vec![exported]);
            let mut graph = graph_with_doc_and_embedding(&[9.0, 9.0]);
            graph.embeddings.clear();
            let before = graph.version();

            let stats = import_embeddings_from_file(
                &mut graph,
                file.path().to_str().unwrap(),
                &RelationshipKeys::new(),
            )
            .unwrap();

            assert_eq!(stats.stores, 0);
            assert_eq!(stats.imported, 0);
            assert_eq!(graph.version(), before);
            assert!(graph.embeddings.is_empty());
        }
    }

    /// A payload whose recorded coverage disagrees with the topology it ships
    /// is refused. A watermark *ahead* of the index would silence a delta that
    /// was never folded in — an index reporting itself current over vectors it
    /// has never seen, which is a wrong answer nothing later notices.
    #[test]
    fn a_payload_whose_watermark_disagrees_with_its_topology_is_skipped() {
        use crate::graph::algorithms::hnsw::{HnswMetric, HnswParams};
        use crate::graph::schema::EmbeddingStore;

        let mut store = EmbeddingStore::new(2);
        for slot in 0..4 {
            store.set_embedding(slot, &[slot as f32, 1.0]);
        }
        let index = HnswIndex::build(
            &store.data,
            &store.norms,
            2,
            HnswMetric::Cosine,
            HnswParams::default(),
            3,
        );
        // `store.norms` is populated by `set_embedding`, so the shape is valid;
        // only the watermark lies.
        let entry = PersistedVectorIndexRef {
            node_type: "Doc",
            embedding_property: "vec_emb",
            index: &index,
            watermark: 4 + 1,
            limit: 1000,
            dirty: Vec::new(),
        };
        let body = codec_ser(serde_codec::CodecVersion::PostcardV1, &vec![entry]).unwrap();
        let mut payload = Vec::new();
        payload.extend_from_slice(VECTOR_INDEX_MAGIC);
        payload.extend_from_slice(&VECTOR_INDEX_FORMAT_VERSION.to_le_bytes());
        payload.extend_from_slice(&body);

        let mut graph = DirGraph::new();
        graph
            .embeddings
            .insert(("Doc".to_string(), "vec_emb".to_string()), store);
        decode_vector_indexes(&payload, &mut graph);
        assert!(
            !graph.embeddings[&("Doc".to_string(), "vec_emb".to_string())].has_index(),
            "a watermark ahead of the topology must be refused"
        );
    }

    /// A node-only export is byte-for-byte the v3 file it always was: header
    /// version 3, then the gzip of the node stores alone. Readers up to 0.17.12
    /// accept exactly these bytes, so a v4 header here would lock them out of
    /// files that carry nothing they cannot read.
    #[test]
    fn export_is_atomic_and_leaves_no_temp_beside_the_file() {
        let graph = graph_with_doc_and_embedding(&[0.25, 0.75]);
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("out.kgle");
        std::fs::write(&dest, b"previous contents").unwrap();
        let path = dest.to_str().unwrap();
        export_embeddings_to_file(&graph, path, None, &RelationshipKeys::new()).unwrap();
        let names: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        assert_eq!(names, vec!["out.kgle".to_string()]);
        assert_eq!(&std::fs::read(&dest).unwrap()[..4], &KGLE_MAGIC);

        // A failed export (unwritable directory) keeps the previous file whole.
        let missing = dir.path().join("no-such-dir").join("out.kgle");
        assert!(export_embeddings_to_file(
            &graph,
            missing.to_str().unwrap(),
            None,
            &RelationshipKeys::new()
        )
        .is_err());
        assert_eq!(&std::fs::read(&dest).unwrap()[..4], &KGLE_MAGIC);
    }

    #[test]
    fn a_node_only_export_writes_the_v3_bytes() {
        let graph = graph_with_doc_and_embedding(&[0.25, 0.75]);
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap();
        let stats =
            export_embeddings_to_file(&graph, path, None, &RelationshipKeys::new()).unwrap();
        assert_eq!(
            (stats.relationship_stores, stats.relationship_embeddings),
            (0, 0)
        );

        let expected_stores = vec![ExportedEmbeddingStore {
            node_type: "Doc".to_string(),
            text_column: "summary".to_string(),
            dimension: 2,
            metric: None,
            model_id: None,
            entries: vec![(Value::UniqueId(7), vec![0.25, 0.75], None)],
        }];
        let payload = codec_ser(serde_codec::CodecVersion::PostcardV1, &expected_stores).unwrap();
        let expected = embedding_file(
            3,
            Some(serde_codec::CodecVersion::PostcardV1.tag()),
            &payload,
        );
        assert_eq!(std::fs::read(path).unwrap(), expected);
    }

    /// The reader matches versions exactly: v3 and v4 decode, a newer file is
    /// refused as newer, and only a genuinely older one gets the pre-0.14
    /// message. A `version < current` rule would call every v3 file pre-0.14.
    #[test]
    fn the_reader_matches_each_version_explicitly() {
        let tag = Some(serde_codec::CodecVersion::PostcardV1.tag());
        let v4 = KgleV4Payload {
            nodes: vec![fixture_store()],
            relationships: Vec::new(),
        };
        let payload = codec_ser(serde_codec::CodecVersion::PostcardV1, &v4).unwrap();
        let decoded = decode_embedding_file_payload(&embedding_file(4, tag, &payload), 4).unwrap();
        assert_eq!(decoded.nodes[0].node_type, "Doc");
        assert!(decoded.relationships.is_empty());

        let newer = decode_embedding_file_payload(&embedding_file(5, tag, &payload), 5)
            .err()
            .unwrap()
            .to_string();
        assert!(
            newer.contains("version 5 is newer than supported version 4"),
            "{newer}"
        );
        let older = decode_embedding_file_payload(&embedding_file(2, tag, &payload), 2)
            .err()
            .unwrap()
            .to_string();
        assert!(older.contains("pre-0.14"), "{older}");
    }

    #[test]
    fn pre_014_embedding_payload_is_rejected() {
        let stores = vec![fixture_store()];
        let old_payload = codec_ser(serde_codec::CodecVersion::PostcardV1, &stores).unwrap();
        let old = embedding_file(2, None, &old_payload);
        let error = decode_embedding_file_payload(&old, 2).err().unwrap();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("pre-0.14"));
    }

    #[test]
    fn postcard_v3_embedding_payload_decodes() {
        let stores = vec![fixture_store()];
        let postcard_payload = codec_ser(serde_codec::CodecVersion::PostcardV1, &stores).unwrap();
        let current = embedding_file(
            3,
            Some(serde_codec::CodecVersion::PostcardV1.tag()),
            &postcard_payload,
        );
        let decoded = decode_embedding_file_payload(&current, 3).unwrap().nodes;
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].node_type, "Doc");
        assert_eq!(decoded[0].text_column, "summary");
        assert_eq!(decoded[0].dimension, 2);
        assert_eq!(decoded[0].metric.as_deref(), Some("cosine"));
        assert_eq!(decoded[0].model_id.as_deref(), Some("fixture"));
        assert_eq!(decoded[0].entries[0].2, Some(99));
    }

    #[test]
    fn postcard_v3_embedding_payload_requires_its_codec_tag() {
        let truncated = [b'K', b'G', b'L', b'E', 3, 0, 0, 0];
        assert!(decode_embedding_file_payload(&truncated, 3)
            .err()
            .unwrap()
            .to_string()
            .contains("codec tag"));

        let invalid = [b'K', b'G', b'L', b'E', 3, 0, 0, 0, 99];
        assert!(decode_embedding_file_payload(&invalid, 3)
            .err()
            .unwrap()
            .to_string()
            .contains("Invalid .kgle codec tag"));
    }
}
