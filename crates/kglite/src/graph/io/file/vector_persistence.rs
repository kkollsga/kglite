//! Vector-cache and standalone embedding-file persistence.

use super::{codec_deser, codec_ser, MAX_CODEC_BYTES};
use crate::datatypes::values::Value;
use crate::graph::algorithms::hnsw::HnswIndex;
use crate::graph::schema::DirGraph;
use crate::graph::storage::GraphRead;
use crate::serde_codec;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};

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
//   [12..]   bincode( Vec<(node_type, embedding_property, HnswIndex)> )
const VECTOR_INDEX_MAGIC: &[u8; 8] = b"KGLVIDX1";
const VECTOR_INDEX_FORMAT_VERSION: u32 = 1;

/// Encode every built HNSW index into a self-describing payload. Returns `None`
/// when no store carries an index (the section is then omitted entirely).
pub(super) fn encode_vector_indexes(graph: &DirGraph) -> io::Result<Option<Vec<u8>>> {
    let entries: Vec<(&String, &String, &HnswIndex)> = graph
        .embeddings
        .iter()
        .filter_map(|((nt, prop), s)| s.index.as_ref().map(|idx| (nt, prop, idx)))
        .collect();
    if entries.is_empty() {
        return Ok(None);
    }
    let body = codec_ser(&entries)?;
    let mut payload = Vec::with_capacity(12 + body.len());
    payload.extend_from_slice(VECTOR_INDEX_MAGIC);
    payload.extend_from_slice(&VECTOR_INDEX_FORMAT_VERSION.to_le_bytes());
    payload.extend_from_slice(&body);
    Ok(Some(payload))
}

/// Decode the vector-index section and attach indexes to the matching stores.
/// Best-effort: an unrecognised magic, an unknown format version, a bincode
/// error, or a shape mismatch against the loaded store all result in the index
/// being silently skipped — never a load failure. Must run AFTER embeddings are
/// loaded and their norms rebuilt (cosine navigation needs the norm cache).
pub(super) fn decode_vector_indexes(payload: &[u8], graph: &mut DirGraph) {
    if payload.len() < 12 || &payload[..8] != VECTOR_INDEX_MAGIC {
        return;
    }
    let ver = u32::from_le_bytes([payload[8], payload[9], payload[10], payload[11]]);
    if ver != VECTOR_INDEX_FORMAT_VERSION {
        return; // newer/older index format — skip, the store rebuilds on demand
    }
    let entries: Vec<(String, String, HnswIndex)> = match codec_deser(&payload[12..]) {
        Ok(e) => e,
        Err(_) => return,
    };
    for (node_type, prop, idx) in entries {
        if let Some(store) = graph.embeddings.get_mut(&(node_type, prop)) {
            // Defensive: only attach an index whose shape still matches the
            // store it was built over (dimension + vector count).
            if idx.dim() == store.dimension && idx.len() == store.len() {
                store.index = Some(idx);
            }
        }
    }
}

// ─── Embedding Export / Import ────────────────────────────────────────────

/// Magic bytes for the embedding export format.
const KGLE_MAGIC: [u8; 4] = *b"KGLE";
/// v2 (0.11.1): carries store `metric` + `model_id` + per-node `text_hash`
/// alongside each vector, so a rebuild-from-`.kgle` pipeline retains provenance
/// and can use `embed_texts(mode='changed')`. v1 files (no provenance) still
/// read — see the version branch in `import_embeddings_from_file`.
const KGLE_VERSION: u32 = 2;

/// v1 `.kgle` store shape (no provenance) — kept solely to deserialize files
/// produced by kglite ≤ 0.11.0.
#[derive(Deserialize)]
struct ExportedEmbeddingStoreV1 {
    node_type: String,
    text_column: String,
    dimension: usize,
    entries: Vec<(Value, Vec<f32>)>,
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
}

/// Export embeddings to a standalone .kgle file, keyed by node ID.
pub fn export_embeddings_to_file(
    graph: &DirGraph,
    path: &str,
    filter: Option<&EmbeddingExportFilter>,
) -> io::Result<ExportStats> {
    // Arena guard: node_weight materializes on the disk backend
    // (protocol in disk/graph.rs); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    let mut exported_stores: Vec<ExportedEmbeddingStore> = Vec::new();
    let mut total_embeddings = 0usize;

    for ((node_type, store_name), store) in &graph.embeddings {
        let text_column = store_name
            .strip_suffix("_emb")
            .unwrap_or(store_name.as_str());

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
                .node_weight(petgraph::graph::NodeIndex::new(node_index))
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

    // Write: magic + version + gzip(bincode(stores))
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&KGLE_MAGIC)?;
    writer.write_all(&KGLE_VERSION.to_le_bytes())?;

    let gz = GzEncoder::new(&mut writer, Compression::new(3));
    serde_codec::encode_into_bounded(gz, &exported_stores, MAX_CODEC_BYTES)
        .map_err(|e| io::Error::other(format!("Failed to serialize embeddings: {}", e)))?;

    writer.flush()?;

    Ok(ExportStats {
        stores: exported_stores.len(),
        embeddings: total_embeddings,
    })
}

/// Import embeddings from a .kgle file, resolving node IDs to current graph indices.
pub fn import_embeddings_from_file(graph: &mut DirGraph, path: &str) -> io::Result<ImportStats> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut buf = Vec::new();
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
    if version > KGLE_VERSION {
        return Err(io::Error::other(format!(
            "Embedding file version {} is newer than supported version {}. Please upgrade kglite.",
            version, KGLE_VERSION,
        )));
    }

    // Decompress and deserialize. bincode is positional, so the v1 shape
    // (no provenance) must be read with its own struct and lifted to v2.
    let gz = GzDecoder::new(&buf[8..]);
    let exported_stores: Vec<ExportedEmbeddingStore> = if version >= 2 {
        serde_codec::decode_from_bounded(gz, MAX_CODEC_BYTES)
            .map_err(|e| io::Error::other(format!("Failed to deserialize embedding data: {}", e)))?
    } else {
        let v1: Vec<ExportedEmbeddingStoreV1> =
            serde_codec::decode_from_bounded(gz, MAX_CODEC_BYTES).map_err(|e| {
                io::Error::other(format!("Failed to deserialize embedding data: {}", e))
            })?;
        v1.into_iter()
            .map(|s| ExportedEmbeddingStore {
                node_type: s.node_type,
                text_column: s.text_column,
                dimension: s.dimension,
                metric: None,
                model_id: None,
                entries: s.entries.into_iter().map(|(id, v)| (id, v, None)).collect(),
            })
            .collect()
    };

    let mut total_imported = 0usize;
    let mut total_skipped = 0usize;
    let mut stores_count = 0usize;
    let mut dropped_stores = 0usize;

    for exported in exported_stores {
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

        for (id, vec, hash) in &exported.entries {
            match graph.lookup_by_id(&exported.node_type, id) {
                Some(node_idx) => {
                    store.set_embedding(node_idx.index(), vec);
                    // Restore the per-node text hash so embed_texts(mode='changed')
                    // can diff against it (the whole point of v2 provenance).
                    if let Some(h) = hash {
                        store.set_text_hash(node_idx.index(), *h);
                    }
                    imported += 1;
                }
                None => {
                    skipped += 1;
                }
            }
        }

        if imported > 0 {
            let key = (exported.node_type, format!("{}_emb", exported.text_column));
            graph.embeddings.insert(key, store);
            stores_count += 1;
        } else if !exported.entries.is_empty() {
            dropped_stores += 1;
        }

        total_imported += imported;
        total_skipped += skipped;
    }

    Ok(ImportStats {
        stores: stores_count,
        imported: total_imported,
        skipped: total_skipped,
        dropped_stores,
    })
}
