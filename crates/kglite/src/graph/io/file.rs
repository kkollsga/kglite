// src/graph/file.rs
//
// Versioned binary format for KnowledgeGraph persistence.
//
// File format v4 layout (Phase A.1 / C5 of bolt_implementation.md):
//   [0..4]    Magic: b"RGF\x04" (Rusty Graph Format, version 4)
//   [4..8]    core_data_version: u32 LE (tracks NodeData/EdgeData/Value changes)
//   [8..12]   metadata_length: u32 LE
//   [12..12+N]  JSON metadata (column schemas, section sizes, all config)
//   [section]  topology.zst — graph structure WITHOUT node properties
//   [section]  columns_<Type>.zst — one per node type, packed column data
//   [section]  embeddings.zst (optional)
//   [section]  timeseries.zst (optional)
//
// v4 vs v3: the Value enum gained five variants (Node, Relationship,
// Path, List, Map). Variants 0..=8 (the v3 scalar set) preserve their
// serde discriminants, so v3 files COULD be deserialised structurally
// — but a v3 binary cannot read v4 files (unknown discriminants 9..=13),
// and Phase A.1 makes the *hard break* user-decision: a v4 binary
// refuses v3 files outright with a clear "rebuild your graph" message.
// One file format, one set of in-flight Value semantics.

use crate::graph::features::timeseries::{NodeTimeseries, TimeseriesConfig};
use crate::graph::schema::{
    CompositeIndexKey, ConnectionTypeInfo, ConnectivityTriple, DirGraph, EmbeddingStore, IndexKey,
    PropertyStorage, SaveMetadata, SchemaDefinition, SerdeDeserializeGuard, SerdeSerializeGuard,
    SpatialConfig, StringInterner, StripPropertiesGuard, TemporalConfig,
};
use crate::graph::storage::column_store::ColumnStore;
use crate::graph::storage::{GraphRead, GraphWrite};
// This module no longer constructs `KnowledgeGraph` directly.
// `load_file` / `load_disk_dir` / `load_v4` return
// `Arc<DirGraph>`; the binding callsites wrap that in their own
// ergonomic type (pyapi → `KnowledgeGraph`, mcp-server → its
// own `ActiveGraph`, future Go/TS → their binding's struct).
// Keeps io decoupled from binding state.
use bincode::Options;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::sync::Arc;

/// Return a pinned bincode configuration that is identical to the legacy
/// `bincode::serialize` / `bincode::deserialize` encoding:
///   - Fixed-size integer encoding (not varint)
///   - Little-endian byte order
///   - No trailing bytes rejected
///   - 2 GiB size limit (generous, prevents OOM on corrupt files)
///
/// Using explicit options guarantees wire-format stability regardless of
/// bincode crate default changes or future upgrades.
fn bincode_options() -> impl bincode::Options {
    bincode::options()
        .with_fixint_encoding()
        .with_little_endian()
        .allow_trailing_bytes()
        .with_limit(2 * 1024 * 1024 * 1024) // 2 GiB
}

/// Magic bytes for the v3 columnar format: "RGF\x03". Retained ONLY
/// so the loader can detect a v3 file and emit a specific
/// "rebuild your graph" error rather than a generic "unrecognized".
const V3_MAGIC: [u8; 4] = [0x52, 0x47, 0x46, 0x03];

/// Magic bytes for the v4 columnar format: "RGF\x04". Phase A.1 / C5
/// introduced v4 alongside the `Value::Node`/`Relationship`/`Path`/
/// `List`/`Map` enum extension. Hard break on v3 files (no read-compat
/// path) per the bolt_implementation.md plan.
const V4_MAGIC: [u8; 4] = [0x52, 0x47, 0x46, 0x04];

/// Current core data version. Bump ONLY when NodeData, EdgeData, or Value enum changes.
/// This is independent of metadata — metadata uses JSON and handles changes via serde defaults.
///
/// 0.9.52 / Phase A.1: bumped to 2 — the `Value` enum gained five
/// structured variants (Node, Relationship, Path, List, Map).
const CURRENT_CORE_DATA_VERSION: u32 = 2;

/// Column section metadata for v3 format.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct V3ColumnSection {
    type_name: String,
    compressed_size: u64,
    row_count: u32,
    columns: HashMap<String, String>, // prop_name → type_tag
}

/// Metadata serialized as JSON in v3 files. All fields use `#[serde(default)]`
/// so that adding/removing fields never breaks existing files.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct FileMetadata {
    /// Core data version at save time — must match or be migratable.
    #[serde(default)]
    core_data_version: u32,
    /// Library version string at save time (e.g. "0.6.5").
    #[serde(default)]
    library_version: String,
    /// Optional schema definition.
    #[serde(default)]
    schema_definition: Option<SchemaDefinition>,
    /// Property index keys to rebuild after load.
    #[serde(default)]
    property_index_keys: Vec<IndexKey>,
    /// Composite index keys to rebuild after load.
    #[serde(default)]
    composite_index_keys: Vec<CompositeIndexKey>,
    /// Range index keys to rebuild after load.
    #[serde(default)]
    range_index_keys: Vec<IndexKey>,
    /// Node type metadata: node_type → { property_name → type_string }
    #[serde(default)]
    node_type_metadata: HashMap<String, HashMap<String, String>>,
    /// Connection type metadata: connection_type → ConnectionTypeInfo
    #[serde(default)]
    connection_type_metadata: HashMap<String, ConnectionTypeInfo>,
    /// Original ID field name per node type (for alias resolution)
    #[serde(default)]
    id_field_aliases: HashMap<String, String>,
    /// Original title field name per node type (for alias resolution)
    #[serde(default)]
    title_field_aliases: HashMap<String, String>,
    /// Auto-vacuum threshold (None = disabled, default Some(0.3))
    #[serde(default = "crate::graph::dir_graph::default_auto_vacuum_threshold")]
    auto_vacuum_threshold: Option<f64>,
    /// Parent types: child_type → parent_type. Determines which types are
    /// "core" vs "supporting" in describe() output.
    #[serde(default)]
    parent_types: HashMap<String, String>,
    /// Spatial configuration per node type.
    #[serde(default)]
    spatial_configs: HashMap<String, SpatialConfig>,
    /// Timeseries configuration per node type.
    #[serde(default)]
    timeseries_configs: HashMap<String, TimeseriesConfig>,
    /// Temporal configuration per node type (valid_from/valid_to on nodes).
    #[serde(default)]
    temporal_node_configs: HashMap<String, TemporalConfig>,
    /// Temporal configuration per connection type (valid_from/valid_to on edges).
    #[serde(default)]
    temporal_edge_configs: HashMap<String, Vec<TemporalConfig>>,
    /// Timeseries data version: 1 = Vec<Vec<i64>> keys (legacy), 2 = NaiveDate keys.
    #[serde(default = "default_ts_data_version")]
    timeseries_data_version: u32,
    /// v3: compressed size of topology section.
    #[serde(default)]
    topology_compressed_size: u64,
    /// v3: column sections metadata (one per node type).
    #[serde(default)]
    column_sections: Vec<V3ColumnSection>,
    /// v3: compressed size of embedding section (0 if none).
    #[serde(default)]
    embeddings_compressed_size: u64,
    /// v3: compressed size of timeseries section (0 if none).
    #[serde(default)]
    timeseries_compressed_size: u64,
    /// 0.10.5: compressed size of secondary-label-index section (0 if
    /// none). Persists `DirGraph.secondary_label_index` for in-memory
    /// graphs. Disk graphs use the parallel `secondary_labels.bin.zst`
    /// sidecar. Older `.kgl` files default to 0 (no section to read).
    #[serde(default)]
    secondary_labels_compressed_size: u64,
    /// Cached edge type counts (connection_type → count).
    /// Persisted from warm cache on save, restored to cache on load.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    edge_type_counts: Option<HashMap<String, usize>>,
    /// Type connectivity triples: (src_type, conn_type, tgt_type, count).
    /// Pre-computed type-level graph for instant describe() at any scale.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    type_connectivity: Option<Vec<ConnectivityTriple>>,
}

fn default_ts_data_version() -> u32 {
    2
}

// ─── Metadata transfer helpers ───────────────────────────────────────────────

impl FileMetadata {
    /// Build metadata from a DirGraph, leaving v3 section sizes at zero
    /// (caller fills them in after compression).
    pub(crate) fn from_graph(graph: &DirGraph) -> Self {
        FileMetadata {
            core_data_version: CURRENT_CORE_DATA_VERSION,
            library_version: env!("CARGO_PKG_VERSION").to_string(),
            schema_definition: graph.schema_definition.clone(),
            property_index_keys: graph.property_index_keys.clone(),
            composite_index_keys: graph.composite_index_keys.clone(),
            range_index_keys: graph.range_index_keys.clone(),
            node_type_metadata: graph.node_type_metadata.clone(),
            connection_type_metadata: graph.connection_type_metadata.clone(),
            id_field_aliases: graph.id_field_aliases.clone(),
            title_field_aliases: graph.title_field_aliases.clone(),
            auto_vacuum_threshold: graph.auto_vacuum_threshold,
            parent_types: graph.parent_types.clone(),
            spatial_configs: graph.spatial_configs.clone(),
            timeseries_configs: graph.timeseries_configs.clone(),
            temporal_node_configs: graph.temporal_node_configs.clone(),
            temporal_edge_configs: graph.temporal_edge_configs.clone(),
            timeseries_data_version: 2,
            // Section sizes filled in by caller:
            topology_compressed_size: 0,
            column_sections: Vec::new(),
            embeddings_compressed_size: 0,
            timeseries_compressed_size: 0,
            secondary_labels_compressed_size: 0,
            // Persist edge type counts if cache is warm (no O(E) scan if cold)
            edge_type_counts: if graph.has_edge_type_counts_cache() {
                Some(graph.get_edge_type_counts())
            } else {
                None
            },
            // Persist type connectivity if computed.
            // 0.8.13: `DirGraph::save_disk` strips this field from the
            // disk-mode metadata.json and writes
            // `type_connectivity.bin.zst` separately (3.17 M-entry JSON
            // list → packed binary). In-memory .kgl saves keep embedding
            // it here for single-file portability.
            type_connectivity: graph.get_type_connectivity(),
        }
    }

    /// Apply metadata fields to a DirGraph during load. Equivalent to
    /// `apply_to_with(graph, true)` — preserved for the in-memory `.kgl`
    /// load path that doesn't have a separate `type_connectivity.bin.zst`.
    #[allow(dead_code)]
    pub(crate) fn apply_to(self, graph: &mut DirGraph) {
        self.apply_to_with(graph, true)
    }

    /// Apply metadata fields with control over the type-connectivity
    /// derive fallback. Disk loaders pass `derive_type_connectivity=false`
    /// when a dedicated `type_connectivity.bin.zst` will populate the
    /// cache below — the cartesian-product derive over
    /// `connection_type_metadata` clones millions of String triples on
    /// large graphs and dominated load time before this gate.
    pub(crate) fn apply_to_with(self, graph: &mut DirGraph, derive_type_connectivity: bool) {
        graph.schema_definition = self.schema_definition;
        graph.property_index_keys = self.property_index_keys;
        graph.composite_index_keys = self.composite_index_keys;
        graph.range_index_keys = self.range_index_keys;
        graph.node_type_metadata = self.node_type_metadata;
        graph.connection_type_metadata = self.connection_type_metadata;
        graph.id_field_aliases = self.id_field_aliases;
        graph.title_field_aliases = self.title_field_aliases;
        graph.auto_vacuum_threshold = self.auto_vacuum_threshold;
        graph.parent_types = self.parent_types;
        graph.spatial_configs = self.spatial_configs;
        graph.timeseries_configs = self.timeseries_configs;
        graph.temporal_node_configs = self.temporal_node_configs;
        graph.temporal_edge_configs = self.temporal_edge_configs;
        graph.save_metadata = SaveMetadata {
            format_version: 3,
            library_version: self.library_version,
        };
        // Restore edge type counts cache if persisted
        if let Some(counts) = self.edge_type_counts {
            *graph.edge_type_counts_cache.write().unwrap() = Some(counts);
        }
        // Restore type connectivity cache if persisted
        if let Some(triples) = self.type_connectivity {
            *graph.type_connectivity_cache.write().unwrap() = Some(triples);
        } else if derive_type_connectivity && !graph.connection_type_metadata.is_empty() {
            // Derive type connectivity from connection_type_metadata (instant, no I/O).
            // This covers older graphs that don't have persisted type_connectivity.
            let edge_counts = graph.edge_type_counts_cache.read().unwrap();
            let mut triples = Vec::new();
            for (conn_type, info) in &graph.connection_type_metadata {
                let count = edge_counts
                    .as_ref()
                    .and_then(|c| c.get(conn_type).copied())
                    .unwrap_or(0);
                for src in &info.source_types {
                    for tgt in &info.target_types {
                        triples.push(crate::graph::schema::ConnectivityTriple {
                            src: src.clone(),
                            conn: conn_type.clone(),
                            tgt: tgt.clone(),
                            count,
                        });
                    }
                }
            }
            if !triples.is_empty() {
                *graph.type_connectivity_cache.write().unwrap() = Some(triples);
            }
        }
    }
}

/// Build metadata for disk-mode save (reuses the same FileMetadata structure).
pub(crate) fn build_disk_metadata(graph: &DirGraph) -> FileMetadata {
    FileMetadata::from_graph(graph)
}

/// Strip `type_connectivity` from FileMetadata so the disk-mode save
/// path can emit it into `type_connectivity.bin.zst` instead. The
/// in-memory `.kgl` save path keeps the embedded form.
pub(crate) fn strip_type_connectivity(meta: &mut FileMetadata) {
    meta.type_connectivity = None;
}

/// Strip the two heavy HashMap fields from FileMetadata so the disk-mode
/// save path can emit them into dedicated binary sidecars. On
/// slice-built Wikidata graphs with 30K-50K node types, parsing these
/// fields out of `metadata.json` cost 4-5 seconds; the binary form
/// loads in <100 ms.
pub(crate) fn strip_heavy_metadata(meta: &mut FileMetadata) {
    meta.node_type_metadata.clear();
    meta.connection_type_metadata.clear();
}

// ─── node_type_metadata.bin.zst (0.8.28 fast-load) ───────────────────────────
//
// Replaces ~50% of `metadata.json` parse cost on slice-built graphs. The
// field is HashMap<String, HashMap<String, String>> = {type_name:
// {prop_name: prop_type_str}}. JSON parses 50K outer × 3 inner entries
// in ~2 s; the packed binary parses the same payload in <50 ms.
//
// Payload (pre-zstd):
//   [ 0.. 8]  magic       = b"KGLNTM1\0"
//   [ 8..12]  version     = u32 LE (= 1)
//   [12..16]  num_types   = u32 LE
//   per type (repeated num_types times):
//     name_len:    u32 LE
//     name:        [u8; name_len]   (UTF-8)
//     num_props:   u32 LE
//     per prop:
//       prop_name_len: u32 LE
//       prop_name:     [u8; prop_name_len]   (UTF-8)
//       prop_type_len: u32 LE
//       prop_type:     [u8; prop_type_len]   (UTF-8)

const NODE_TYPE_META_MAGIC: &[u8; 8] = b"KGLNTM1\0";
const NODE_TYPE_META_VERSION: u32 = 1;

pub(crate) fn write_node_type_metadata_bin(
    dir: &std::path::Path,
    graph: &DirGraph,
) -> Result<(), String> {
    if graph.node_type_metadata.is_empty() {
        return Ok(());
    }

    // Sort entries deterministically so re-saves produce byte-identical
    // files for clean diffs.
    let mut entries: Vec<(&String, &HashMap<String, String>)> =
        graph.node_type_metadata.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));

    let mut payload: Vec<u8> = Vec::with_capacity(64 * 1024);
    payload.extend_from_slice(NODE_TYPE_META_MAGIC);
    payload.extend_from_slice(&NODE_TYPE_META_VERSION.to_le_bytes());
    payload.extend_from_slice(&(entries.len() as u32).to_le_bytes());

    for (type_name, props) in entries {
        payload.extend_from_slice(&(type_name.len() as u32).to_le_bytes());
        payload.extend_from_slice(type_name.as_bytes());

        let mut prop_pairs: Vec<(&String, &String)> = props.iter().collect();
        prop_pairs.sort_by(|a, b| a.0.cmp(b.0));
        payload.extend_from_slice(&(prop_pairs.len() as u32).to_le_bytes());
        for (k, v) in prop_pairs {
            payload.extend_from_slice(&(k.len() as u32).to_le_bytes());
            payload.extend_from_slice(k.as_bytes());
            payload.extend_from_slice(&(v.len() as u32).to_le_bytes());
            payload.extend_from_slice(v.as_bytes());
        }
    }

    let compressed = zstd::encode_all(payload.as_slice(), 3)
        .map_err(|e| format!("node_type_metadata compression failed: {}", e))?;
    std::fs::write(dir.join("node_type_metadata.bin.zst"), compressed)
        .map_err(|e| format!("Failed to write node_type_metadata.bin.zst: {}", e))?;
    Ok(())
}

pub(crate) fn read_node_type_metadata_bin(
    dir: &std::path::Path,
) -> io::Result<Option<HashMap<String, HashMap<String, String>>>> {
    let path = dir.join("node_type_metadata.bin.zst");
    if !path.exists() {
        return Ok(None);
    }
    let compressed = std::fs::read(&path)?;
    let bytes = zstd::decode_all(compressed.as_slice()).map_err(io::Error::other)?;
    if bytes.len() < 16 || &bytes[..8] != NODE_TYPE_META_MAGIC {
        return Ok(None);
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if version != NODE_TYPE_META_VERSION {
        return Ok(None);
    }
    let num_types = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;

    let mut out = HashMap::with_capacity(num_types);
    let mut cursor = 16usize;
    for _ in 0..num_types {
        let name = read_lp_string(&bytes, &mut cursor)?;
        let num_props = read_u32(&bytes, &mut cursor)? as usize;
        let mut props = HashMap::with_capacity(num_props);
        for _ in 0..num_props {
            let k = read_lp_string(&bytes, &mut cursor)?;
            let v = read_lp_string(&bytes, &mut cursor)?;
            props.insert(k, v);
        }
        out.insert(name, props);
    }
    Ok(Some(out))
}

// ─── connection_type_metadata.bin.zst (0.8.28 fast-load) ──────────────────────
//
// Replaces ~40% of `metadata.json` parse cost. Field is
// HashMap<String, ConnectionTypeInfo>. ConnectionTypeInfo carries
// source_types/target_types HashSets plus a property_types map.
//
// Payload (pre-zstd):
//   [ 0.. 8]  magic       = b"KGLCTM1\0"
//   [ 8..12]  version     = u32 LE (= 1)
//   [12..16]  num_conns   = u32 LE
//   per conn (repeated num_conns times):
//     name_len:    u32, name: [u8]
//     num_sources: u32, then (name_len + name) × num_sources
//     num_targets: u32, then (name_len + name) × num_targets
//     num_props:   u32, then (k_len + k + v_len + v) × num_props

const CONN_TYPE_META_MAGIC: &[u8; 8] = b"KGLCTM1\0";
const CONN_TYPE_META_VERSION: u32 = 1;

pub(crate) fn write_connection_type_metadata_bin(
    dir: &std::path::Path,
    graph: &DirGraph,
) -> Result<(), String> {
    use crate::graph::schema::ConnectionTypeInfo;
    if graph.connection_type_metadata.is_empty() {
        return Ok(());
    }

    let mut entries: Vec<(&String, &ConnectionTypeInfo)> =
        graph.connection_type_metadata.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));

    let mut payload: Vec<u8> = Vec::with_capacity(64 * 1024);
    payload.extend_from_slice(CONN_TYPE_META_MAGIC);
    payload.extend_from_slice(&CONN_TYPE_META_VERSION.to_le_bytes());
    payload.extend_from_slice(&(entries.len() as u32).to_le_bytes());

    for (conn_name, info) in entries {
        payload.extend_from_slice(&(conn_name.len() as u32).to_le_bytes());
        payload.extend_from_slice(conn_name.as_bytes());

        let mut sources: Vec<&String> = info.source_types.iter().collect();
        sources.sort();
        payload.extend_from_slice(&(sources.len() as u32).to_le_bytes());
        for s in sources {
            payload.extend_from_slice(&(s.len() as u32).to_le_bytes());
            payload.extend_from_slice(s.as_bytes());
        }

        let mut targets: Vec<&String> = info.target_types.iter().collect();
        targets.sort();
        payload.extend_from_slice(&(targets.len() as u32).to_le_bytes());
        for t in targets {
            payload.extend_from_slice(&(t.len() as u32).to_le_bytes());
            payload.extend_from_slice(t.as_bytes());
        }

        let mut props: Vec<(&String, &String)> = info.property_types.iter().collect();
        props.sort_by(|a, b| a.0.cmp(b.0));
        payload.extend_from_slice(&(props.len() as u32).to_le_bytes());
        for (k, v) in props {
            payload.extend_from_slice(&(k.len() as u32).to_le_bytes());
            payload.extend_from_slice(k.as_bytes());
            payload.extend_from_slice(&(v.len() as u32).to_le_bytes());
            payload.extend_from_slice(v.as_bytes());
        }
    }

    let compressed = zstd::encode_all(payload.as_slice(), 3)
        .map_err(|e| format!("connection_type_metadata compression failed: {}", e))?;
    std::fs::write(dir.join("connection_type_metadata.bin.zst"), compressed)
        .map_err(|e| format!("Failed to write connection_type_metadata.bin.zst: {}", e))?;
    Ok(())
}

pub(crate) fn read_connection_type_metadata_bin(
    dir: &std::path::Path,
) -> io::Result<Option<HashMap<String, crate::graph::schema::ConnectionTypeInfo>>> {
    use crate::graph::schema::ConnectionTypeInfo;
    let path = dir.join("connection_type_metadata.bin.zst");
    if !path.exists() {
        return Ok(None);
    }
    let compressed = std::fs::read(&path)?;
    let bytes = zstd::decode_all(compressed.as_slice()).map_err(io::Error::other)?;
    if bytes.len() < 16 || &bytes[..8] != CONN_TYPE_META_MAGIC {
        return Ok(None);
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if version != CONN_TYPE_META_VERSION {
        return Ok(None);
    }
    let num_conns = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;

    let mut out = HashMap::with_capacity(num_conns);
    let mut cursor = 16usize;
    for _ in 0..num_conns {
        let name = read_lp_string(&bytes, &mut cursor)?;
        let num_sources = read_u32(&bytes, &mut cursor)? as usize;
        let mut source_types = std::collections::HashSet::with_capacity(num_sources);
        for _ in 0..num_sources {
            source_types.insert(read_lp_string(&bytes, &mut cursor)?);
        }
        let num_targets = read_u32(&bytes, &mut cursor)? as usize;
        let mut target_types = std::collections::HashSet::with_capacity(num_targets);
        for _ in 0..num_targets {
            target_types.insert(read_lp_string(&bytes, &mut cursor)?);
        }
        let num_props = read_u32(&bytes, &mut cursor)? as usize;
        let mut property_types = HashMap::with_capacity(num_props);
        for _ in 0..num_props {
            let k = read_lp_string(&bytes, &mut cursor)?;
            let v = read_lp_string(&bytes, &mut cursor)?;
            property_types.insert(k, v);
        }
        out.insert(
            name,
            ConnectionTypeInfo {
                source_types,
                target_types,
                property_types,
            },
        );
    }
    Ok(Some(out))
}

// Helpers for length-prefixed string + u32 reads.
#[inline]
fn read_u32(bytes: &[u8], cursor: &mut usize) -> io::Result<u32> {
    if *cursor + 4 > bytes.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "metadata sidecar truncated",
        ));
    }
    let v = u32::from_le_bytes(bytes[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    Ok(v)
}

#[inline]
fn read_lp_string(bytes: &[u8], cursor: &mut usize) -> io::Result<String> {
    let len = read_u32(bytes, cursor)? as usize;
    if *cursor + len > bytes.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "metadata sidecar string truncated",
        ));
    }
    let s = std::str::from_utf8(&bytes[*cursor..*cursor + len])
        .map_err(io::Error::other)?
        .to_string();
    *cursor += len;
    Ok(s)
}

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

const TYPE_INDICES_MAGIC: &[u8; 8] = b"KGLTIDX1";
const TYPE_INDICES_VERSION: u32 = 1;

/// Reader for `type_indices.bin.zst` in the new flat-CSR format.
/// Returns `Ok(None)` if the payload does not start with the
/// `KGLTIDX1` magic (caller falls back to the legacy bincode path).
pub(crate) fn read_type_indices_bin(
    payload: &[u8],
    interner: &crate::graph::storage::interner::StringInterner,
) -> io::Result<Option<std::collections::HashMap<String, Vec<petgraph::graph::NodeIndex>>>> {
    if payload.len() < 24 || &payload[..8] != TYPE_INDICES_MAGIC {
        return Ok(None);
    }
    let version = u32::from_le_bytes(payload[8..12].try_into().unwrap());
    if version != TYPE_INDICES_VERSION {
        return Ok(None);
    }
    let num_types = u32::from_le_bytes(payload[12..16].try_into().unwrap()) as usize;
    let total_nodes = u64::from_le_bytes(payload[16..24].try_into().unwrap()) as usize;

    let type_keys_offset = 24usize;
    let offsets_offset = type_keys_offset + 8 * num_types;
    let nodes_offset = offsets_offset + 8 * (num_types + 1);
    let expected_len = nodes_offset + 4 * total_nodes;
    if payload.len() < expected_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "type_indices.bin.zst is truncated",
        ));
    }

    let mut out =
        std::collections::HashMap::<String, Vec<petgraph::graph::NodeIndex>>::with_capacity(
            num_types,
        );
    for i in 0..num_types {
        let tkey_base = type_keys_offset + i * 8;
        let type_key = u64::from_le_bytes(payload[tkey_base..tkey_base + 8].try_into().unwrap());
        let off_base = offsets_offset + i * 8;
        let off_start =
            u64::from_le_bytes(payload[off_base..off_base + 8].try_into().unwrap()) as usize;
        let off_end =
            u64::from_le_bytes(payload[off_base + 8..off_base + 16].try_into().unwrap()) as usize;
        let name = match interner.try_resolve(crate::graph::schema::InternedKey::from_u64(type_key))
        {
            Some(s) => s.to_string(),
            None => continue, // missing interner entry; skip rather than fail
        };
        let nodes_start = nodes_offset + off_start * 4;
        let nodes_end = nodes_offset + off_end * 4;
        let mut vec = Vec::with_capacity(off_end - off_start);
        for chunk in payload[nodes_start..nodes_end].chunks_exact(4) {
            let idx = u32::from_le_bytes(chunk.try_into().unwrap()) as usize;
            vec.push(petgraph::graph::NodeIndex::new(idx));
        }
        out.insert(name, vec);
    }
    Ok(Some(out))
}

// ─── interner.bin.zst (0.8.13 fast-load) ─────────────────────────────────────
//
// Replaces `interner.json` (a `HashMap<String, String>` of
// hash-to-original) with bincode-serialised `Vec<String>` of the
// original strings, zstd-compressed. The hash is re-derived on load
// by `interner.get_or_intern` — FNV of the string is deterministic.
// Dropping the hash halves the on-disk size and eliminates JSON
// parse overhead.

pub(crate) fn write_interner_bin(dir: &std::path::Path, graph: &DirGraph) -> Result<(), String> {
    let originals: Vec<String> = graph.interner.iter().map(|(_, v)| v.to_string()).collect();
    let bytes = bincode::serialize(&originals)
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
    let bytes = zstd::decode_all(compressed.as_slice()).map_err(io::Error::other)?;
    let originals: Vec<String> = bincode::deserialize(&bytes).map_err(io::Error::other)?;
    for s in &originals {
        graph.interner.get_or_intern(s);
    }
    Ok(true)
}

// ─── id_indices.bin.zst legacy reader ─────────────────────────────────────
//
// Read-only fallback for graphs saved by 0.8.13–0.8.27. Fresh saves use
// the mmap-resident `id_indices.bin` raw layout from
// `storage/disk/id_index.rs::write_id_indices_bin`.

const ID_INDICES_MAGIC: &[u8; 8] = b"KGLIIDX1";
const ID_INDICES_VERSION: u32 = 1;

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
        return Ok(None);
    }
    let num_types = u32::from_le_bytes(payload[12..16].try_into().unwrap()) as usize;
    let mut out = std::collections::HashMap::<String, TypeIdIndex>::with_capacity(num_types);

    let mut cursor = 16usize;
    for _ in 0..num_types {
        if cursor + 24 > payload.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "id_indices.bin.zst truncated at block header",
            ));
        }
        let type_key = u64::from_le_bytes(payload[cursor..cursor + 8].try_into().unwrap());
        let variant_tag = payload[cursor + 8];
        // payload[cursor+9..cursor+16] is padding (7 bytes).
        let num_entries =
            u64::from_le_bytes(payload[cursor + 16..cursor + 24].try_into().unwrap()) as usize;
        cursor += 24;

        let name = interner
            .try_resolve(crate::graph::schema::InternedKey::from_u64(type_key))
            .map(|s| s.to_string());

        match variant_tag {
            0 => {
                let keys_size = 4 * num_entries;
                let idxs_size = 4 * num_entries;
                if cursor + keys_size + idxs_size > payload.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "id_indices Integer block truncated",
                    ));
                }
                let keys_bytes = &payload[cursor..cursor + keys_size];
                let idxs_bytes = &payload[cursor + keys_size..cursor + keys_size + idxs_size];
                cursor += keys_size + idxs_size;
                if let Some(name) = name {
                    let mut map =
                        std::collections::HashMap::<u32, petgraph::graph::NodeIndex>::with_capacity(
                            num_entries,
                        );
                    for i in 0..num_entries {
                        let k =
                            u32::from_le_bytes(keys_bytes[i * 4..i * 4 + 4].try_into().unwrap());
                        let v = u32::from_le_bytes(idxs_bytes[i * 4..i * 4 + 4].try_into().unwrap())
                            as usize;
                        map.insert(k, petgraph::graph::NodeIndex::new(v));
                    }
                    out.insert(name, TypeIdIndex::Integer(map));
                }
            }
            1 => {
                if cursor + 8 > payload.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "id_indices General block missing blob length",
                    ));
                }
                let blob_len =
                    u64::from_le_bytes(payload[cursor..cursor + 8].try_into().unwrap()) as usize;
                cursor += 8;
                if cursor + blob_len > payload.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "id_indices General blob truncated",
                    ));
                }
                let blob = &payload[cursor..cursor + blob_len];
                cursor += blob_len;
                if let Some(name) = name {
                    let inner: std::collections::HashMap<
                        crate::datatypes::values::Value,
                        petgraph::graph::NodeIndex,
                    > = bincode::deserialize(blob).map_err(io::Error::other)?;
                    let _ = num_entries; // redundant with inner.len(), kept for format symmetry
                    out.insert(name, TypeIdIndex::General(inner));
                }
            }
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("id_indices unknown variant tag {}", other),
                ));
            }
        }
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
// for disk-backed graphs. Memory + mapped backends carry secondaries
// inline on NodeData via bincode; disk's columnar layout has no
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
        let src_key = interner.get_or_intern(&t.src).as_u64();
        let conn_key = interner.get_or_intern(&t.conn).as_u64();
        let tgt_key = interner.get_or_intern(&t.tgt).as_u64();
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
/// to the legacy JSON path).
pub(crate) fn read_type_connectivity_bin(
    dir: &std::path::Path,
    graph: &DirGraph,
) -> io::Result<Option<Vec<crate::graph::schema::ConnectivityTriple>>> {
    let path = dir.join("type_connectivity.bin.zst");
    if !path.exists() {
        return Ok(None);
    }
    let compressed = std::fs::read(&path)?;
    let payload = zstd::decode_all(compressed.as_slice()).map_err(io::Error::other)?;
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
fn encode_secondary_label_index(graph: &DirGraph) -> Option<Vec<u8>> {
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
fn decode_secondary_label_index(payload: &[u8], graph: &mut DirGraph) -> io::Result<bool> {
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
        let key = graph.interner.get_or_intern(&name);
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
    for bucket in index.values_mut() {
        bucket.retain(|idx| graph.graph.node_weight(*idx).is_some());
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
/// don't have it).
pub(crate) fn read_secondary_labels_bin(
    dir: &std::path::Path,
    graph: &mut DirGraph,
) -> io::Result<bool> {
    let path = dir.join("secondary_labels.bin.zst");
    if !path.exists() {
        return Ok(false);
    }
    let compressed = std::fs::read(&path)?;
    let payload = zstd::decode_all(compressed.as_slice()).map_err(io::Error::other)?;
    decode_secondary_label_index(&payload, graph)
}

// ─── Save ────────────────────────────────────────────────────────────────────

/// Stamp save metadata and snapshot index keys. Quick, runs with GIL held.
pub fn prepare_save(graph: &mut Arc<DirGraph>) {
    let g = Arc::make_mut(graph);
    g.save_metadata = SaveMetadata::current();
    g.populate_index_keys();
}

/// Compress data using zstd (level 1 — fastest with good ratio).
fn zstd_compress(data: &[u8]) -> io::Result<Vec<u8>> {
    zstd::encode_all(std::io::Cursor::new(data), 1)
}

/// Decompress zstd-compressed data.
fn zstd_decompress(data: &[u8]) -> io::Result<Vec<u8>> {
    zstd::decode_all(std::io::Cursor::new(data))
}

/// Serialize a value using the project's pinned bincode options.
fn bincode_ser<T: Serialize>(val: &T) -> io::Result<Vec<u8>> {
    bincode_options().serialize(val).map_err(io::Error::other)
}

/// Deserialize a value using the project's pinned bincode options.
fn bincode_deser<'a, T: Deserialize<'a>>(buf: &'a [u8]) -> io::Result<T> {
    bincode_options()
        .deserialize(buf)
        .map_err(|e| io::Error::other(format!("bincode deserialization failed: {}", e)))
}

/// Debug-only: verify every InternedKey in `graph.column_stores`'s schemas
/// resolves to a string in `graph.interner`. Catches the class of bug where
/// a writer synthesizes a key via `InternedKey::from_str()` (just hashing)
/// and mutates a ColumnStore without first calling `interner.get_or_intern()`
/// — `save()` would then serialize the unregistered key and `load()` would
/// see "<unknown>" property names, silently corrupting the data.
///
/// Surfaced by the 0.8.39 SET master-path bug (now fixed). Locked in here
/// so any future regression of the same shape (in this or any other write
/// path) panics loudly in debug builds rather than landing as silent data
/// loss in release.
#[cfg(debug_assertions)]
fn debug_assert_column_keys_registered(graph: &DirGraph) {
    for (type_name, store) in &graph.column_stores {
        let schema = store.schema();
        for (_slot, key) in schema.iter() {
            assert!(
                graph.interner.try_resolve(key).is_some(),
                "kglite invariant violation: ColumnStore for type '{}' contains \
                 InternedKey {} but the source string is not registered in \
                 graph.interner. A writer synthesized the key via \
                 `InternedKey::from_str()` without first calling \
                 `interner.get_or_intern(...)`. save() would silently corrupt \
                 the data — failing fast here so the offending writer is \
                 caught at write time, not at load time on the user's machine.",
                type_name,
                key.as_u64()
            );
        }
    }
}

/// Serialize, compress, and write graph data to v3 file. Heavy I/O, safe to run without GIL.
///
/// The graph MUST have columnar storage enabled before calling this function.
/// The caller (Python `save()`) handles auto-enable/disable.
pub fn write_graph_v3(graph: &DirGraph, path: &str) -> io::Result<()> {
    #[cfg(debug_assertions)]
    debug_assert_column_keys_registered(graph);

    // 1. Serialize topology with properties stripped (v3: node props are in column sections)
    let topology_raw = {
        let _strip = StripPropertiesGuard::new();
        let _guard = SerdeSerializeGuard::new(&graph.interner);
        bincode_ser(&graph.graph)?
    };
    let topology_compressed = zstd_compress(&topology_raw)?;
    drop(topology_raw); // free before compressing columns

    // 2. Serialize column sections (one per node type).
    //
    // Iterate column_stores in sorted order by type_name. `graph.column_stores`
    // is a HashMap whose per-instance RandomState would otherwise cause the
    // section order to vary across processes — breaking byte-level reproducibility
    // that the Phase 4 golden-hash test relies on. Sorting here is free
    // (type_name count is small) and doesn't affect the format: each section
    // is self-describing and the decoder iterates column_sections_meta in order.
    let mut column_sections_meta: Vec<V3ColumnSection> = Vec::new();
    let mut column_sections_data: Vec<Vec<u8>> = Vec::new();

    let mut column_stores_sorted: Vec<(&String, &Arc<ColumnStore>)> =
        graph.column_stores.iter().collect();
    column_stores_sorted.sort_by(|a, b| a.0.cmp(b.0));
    for (type_name, store) in column_stores_sorted {
        let packed = store.write_packed(&graph.interner)?;
        let compressed = zstd_compress(&packed)?;
        drop(packed); // free uncompressed before next type

        // Build column schema
        let mut cols = HashMap::new();
        for (slot, ik) in store.schema().iter() {
            let prop_name = graph.interner.resolve(ik);
            if let Some(col) = store.columns_ref().get(slot as usize) {
                cols.insert(prop_name.to_string(), col.type_tag().to_string());
            }
        }

        column_sections_meta.push(V3ColumnSection {
            type_name: type_name.clone(),
            compressed_size: compressed.len() as u64,
            row_count: store.row_count(),
            columns: cols,
        });
        column_sections_data.push(compressed);
    }

    // 3. Compress embeddings if any
    let embedding_compressed = if !graph.embeddings.is_empty() {
        let raw = bincode_ser(&graph.embeddings)?;
        Some(zstd_compress(&raw)?)
    } else {
        None
    };

    // 4. Compress timeseries if any
    let timeseries_compressed = if !graph.timeseries_store.is_empty() {
        let raw = bincode_ser(&graph.timeseries_store)?;
        Some(zstd_compress(&raw)?)
    } else {
        None
    };

    // 4b. Compress secondary-label index if any. Hand-rolled binary
    // format (encode_secondary_label_index) — InternedKey doesn't
    // derive serde, and the same layout is reused by the disk
    // sidecar (`secondary_labels.bin.zst`).
    let secondary_labels_compressed = match encode_secondary_label_index(graph) {
        Some(payload) => Some(zstd_compress(&payload)?),
        None => None,
    };

    // 5. Build metadata (common fields from graph, then fill in section sizes)
    let mut metadata = FileMetadata::from_graph(graph);
    metadata.topology_compressed_size = topology_compressed.len() as u64;
    metadata.column_sections = column_sections_meta;
    metadata.embeddings_compressed_size = embedding_compressed
        .as_ref()
        .map(|b| b.len() as u64)
        .unwrap_or(0);
    metadata.timeseries_compressed_size = timeseries_compressed
        .as_ref()
        .map(|b| b.len() as u64)
        .unwrap_or(0);
    metadata.secondary_labels_compressed_size = secondary_labels_compressed
        .as_ref()
        .map(|b| b.len() as u64)
        .unwrap_or(0);

    // Canonical JSON: round-trip through serde_json::Value so that all
    // HashMap<String, T> fields (nested at any depth) emit with sorted keys.
    // serde_json::Value::Object is backed by BTreeMap<String, Value> (default
    // feature set), so to_value sorts object keys and to_vec walks the tree
    // in sorted order. Prevents per-process HashMap-randomization from
    // producing different save bytes for the same graph — the byte-level
    // tripwire in `tests/test_phase4_parity.py` depends on this.
    let metadata_value = serde_json::to_value(&metadata).map_err(io::Error::other)?;
    let metadata_json = serde_json::to_vec(&metadata_value).map_err(io::Error::other)?;

    // 6. Write file
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Header: magic (4B) + core_data_version (4B) + metadata_length (4B)
    // Phase A.1 / C5 — write the v4 magic. v3 files become unloadable
    // with this binary (intentional hard break).
    writer.write_all(&V4_MAGIC)?;
    writer.write_all(&CURRENT_CORE_DATA_VERSION.to_le_bytes())?;
    writer.write_all(&(metadata_json.len() as u32).to_le_bytes())?;
    writer.write_all(&metadata_json)?;

    // Topology section
    writer.write_all(&topology_compressed)?;

    // Column sections (one per node type, in metadata order)
    for section_data in &column_sections_data {
        writer.write_all(section_data)?;
    }

    // Embeddings section
    if let Some(emb_data) = &embedding_compressed {
        writer.write_all(emb_data)?;
    }

    // Timeseries section
    if let Some(ts_data) = &timeseries_compressed {
        writer.write_all(ts_data)?;
    }

    // Secondary-label-index section (0.10.5+). Single-label graphs
    // skip this entirely (encode returned None).
    if let Some(sl_data) = &secondary_labels_compressed {
        writer.write_all(sl_data)?;
    }

    writer.flush()?;
    Ok(())
}

/// In-memory save composing `prepare_save` + `enable_columnar` +
/// `write_graph_v3`. Public so non-pyo3 consumers (e.g.
/// `kglite-mcp-server`) can save in-memory graphs without
/// duplicating the dispatch logic from
/// `KnowledgeGraph::save` at `src/graph/pyapi/kg_core.rs`.
///
/// Callers under the GIL should release it around `write_graph_v3`
/// for parallelism with other Python threads — see `kg_core.rs::save`
/// for the canonical split. Rust-only callers (no GIL) just call
/// this directly.
pub fn save_inmemory(graph: &mut Arc<DirGraph>, path: &str) -> io::Result<()> {
    prepare_save(graph);
    {
        let dir = Arc::make_mut(graph);
        dir.enable_columnar();
    }
    write_graph_v3(graph, path)
}

/// Mode-aware save: dispatches to `DirGraph::save_disk` for
/// disk-backed graphs, `save_inmemory` otherwise. Mirrors the
/// dispatch in `KnowledgeGraph::save` at
/// `src/graph/pyapi/kg_core.rs`; both consumers (the pyo3 wrapper
/// and the `kglite-mcp-server` binary) call this so dispatch
/// behaviour can't drift between paths.
pub fn save_graph(graph: &mut Arc<DirGraph>, path: &str) -> Result<(), String> {
    if graph.graph.is_disk() {
        let dir = Arc::make_mut(graph);
        return dir.save_disk(path);
    }
    save_inmemory(graph, path).map_err(|e| e.to_string())
}

// ─── Load ────────────────────────────────────────────────────────────────────

/// Minimum file size to use mmap for the initial file read.
/// Below this threshold, `std::fs::read()` is faster (avoids mmap syscall overhead).
const FILE_MMAP_THRESHOLD: u64 = 65_536; // 64 KB

pub fn load_file(path: &str) -> io::Result<Arc<DirGraph>> {
    // If path is a directory, load as disk graph
    let p = std::path::Path::new(path);
    if p.is_dir() {
        return load_disk_dir(p);
    }

    let file = File::open(path)?;
    let file_len = file.metadata()?.len();

    // For large files, mmap avoids the full copy into a Vec<u8>
    if file_len >= FILE_MMAP_THRESHOLD {
        // SAFETY: `Mmap::map` is unsafe because a concurrent writer could
        // race with the reader. The caller of `load_kgl` is the KGLite
        // Python binding, which holds the GIL; no other process is
        // expected to mutate the file during load.
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < 4 {
            return Err(io::Error::other(
                "File is too small to be a valid kglite file.",
            ));
        }
        if mmap[..4] == V4_MAGIC {
            return load_v4(&mmap);
        }
        if mmap[..4] == V3_MAGIC {
            return Err(io::Error::other(V3_HARD_BREAK_MSG));
        }
        return Err(io::Error::other(
            "Unrecognized file format. This file was saved with an older version of kglite. \
             Please rebuild the graph with the current version and save again.",
        ));
    }

    // Small files: direct read is faster
    let buf = std::fs::read(path)?;
    if buf.len() < 4 {
        return Err(io::Error::other(
            "File is too small to be a valid kglite file.",
        ));
    }
    if buf[..4] == V4_MAGIC {
        load_v4(&buf)
    } else if buf[..4] == V3_MAGIC {
        Err(io::Error::other(V3_HARD_BREAK_MSG))
    } else {
        Err(io::Error::other(
            "Unrecognized file format. This file was saved with an older version of kglite. \
             Please rebuild the graph with the current version and save again.",
        ))
    }
}

/// Hard-break message for v3 files in a v4 binary. Per the
/// Phase A.1 user-decision in bolt_implementation.md: no read-compat
/// path; rebuild the graph from source. Message gives the operator
/// enough breadcrumbs to know what changed and what to do.
const V3_HARD_BREAK_MSG: &str = "kglite .kgl file format v3 is not supported by this binary. \
     kglite 0.10+ uses v4 — the Value enum gained structured Node / \
     Relationship / Path / List / Map variants, which changes the \
     serialised property representation. Rebuild your graph from its \
     original source (CSV, DataFrame, dataset loader) and save again, \
     or downgrade kglite to the 0.9.x line if you need to read this \
     file.";

/// Load a disk-mode graph from a directory.
fn load_disk_dir(dir: &std::path::Path) -> io::Result<Arc<DirGraph>> {
    use crate::graph::io::load_timing::{log_stage, stage_timer};
    use crate::graph::schema::GraphBackend;

    let _load_t = stage_timer();

    // Verify this is a disk graph directory
    if !dir.join("disk_graph_meta.json").exists() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Directory does not contain a valid disk graph (missing disk_graph_meta.json)",
        ));
    }

    let mut graph = DirGraph::new();

    // Load DirGraph metadata. The two heavy HashMap fields
    // (`node_type_metadata`, `connection_type_metadata`) come from
    // dedicated binary sidecars (0.8.28+) when present — they cost
    // 4-5 s of JSON parse on slice-built Wikidata graphs with 30K-50K
    // types, vs <100 ms in the binary form. Older graphs keep the
    // fields embedded in metadata.json and are picked up by the
    // standard JSON parse below.
    let t = stage_timer();
    if dir.join("metadata.json").exists() {
        let meta_bytes = std::fs::read(dir.join("metadata.json"))?;
        let mut meta: FileMetadata = serde_json::from_slice(&meta_bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        if let Some(ntm) = read_node_type_metadata_bin(dir)? {
            meta.node_type_metadata = ntm;
        }
        if let Some(ctm) = read_connection_type_metadata_bin(dir)? {
            meta.connection_type_metadata = ctm;
        }
        // Skip the cartesian-product derive of `type_connectivity` at
        // load time — on slice-built graphs with populated source/target
        // sets it clones tens of millions of String triples (4-15 s).
        // The cache is lazy-populated on first `describe()` access via
        // the existing `compute_type_connectivity` fallback (see
        // `introspection/describe.rs`); read sites that miss the cache
        // already fall through to bounded edge scans.
        meta.apply_to_with(&mut graph, false);
    }
    log_stage("metadata_json", t);

    // Load interner. 0.8.13 prefers `interner.bin.zst` (bincode
    // `Vec<String>` + zstd); old `interner.json` is the backward-compat
    // fallback for 0.8.12-and-earlier graphs.
    let t = stage_timer();
    let loaded_from_bin = read_interner_bin(dir, &mut graph)?;
    if !loaded_from_bin && dir.join("interner.json").exists() {
        let interner_str = std::fs::read_to_string(dir.join("interner.json"))?;
        let interner_map: std::collections::HashMap<String, String> =
            serde_json::from_str(&interner_str)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        for original in interner_map.values() {
            graph.interner.get_or_intern(original);
        }
    }
    log_stage("interner_load", t);

    // Load DiskGraph — compressed files decompressed to temp dir, then mmap'd.
    // Interner is passed mutably because legacy format=0 graphs store edge
    // property keys as strings and need to register them on read.
    let t = stage_timer();
    let (disk_graph, temp_dir) =
        crate::graph::storage::disk::graph::DiskGraph::load_from_dir(dir, &mut graph.interner)?;
    log_stage("disk_graph_load", t);
    // Prefetch hot mmap regions (offset arrays + node_slots) into page cache.
    // On macOS, `madvise(MADV_WILLNEED)` synchronously schedules readahead and
    // can block in the syscall even on warm pages — costs ~0.5–1s on the
    // Wikidata graph. Gated by `KGLITE_PREFETCH=1` so callers that want the
    // first-query latency benefit can opt in. Default off.
    if std::env::var_os("KGLITE_PREFETCH").is_some() {
        let t = stage_timer();
        disk_graph.prefetch_hot_regions();
        log_stage("prefetch_hot_regions", t);
    }
    // Phase 5: this is the `.kgl` → `KnowledgeGraph` construction boundary;
    // assembling the backend variant here is analogous to the PyO3 boundary
    // the storage refactor exempts. Stays as an enum literal.
    graph.graph = GraphBackend::Disk(Box::new(disk_graph));

    // Register temp dir for cleanup on drop
    if let Ok(mut dirs) = graph.temp_dirs.lock() {
        dirs.push(temp_dir);
    }

    // Load type_indices from disk, or rebuild from node_slots if file missing.
    //
    // Format priority:
    //   1. type_indices.bin   — 0.8.28+ raw mmap-resident layout (lazy reads).
    //   2. type_indices.bin.zst with KGLTIDX1 magic — 0.8.13 flat-CSR (eager).
    //   3. type_indices.bin.zst as bincode HashMap — pre-0.8.13 (oldest).
    //   4. node_slots scan fallback for graphs missing the file entirely.
    let t = stage_timer();
    if let GraphBackend::Disk(ref dg) = graph.graph {
        let mut loaded = false;
        if let Some(base) =
            crate::graph::storage::disk::type_index::TypeIndexBase::load_from(dir, &graph.interner)?
        {
            graph.type_indices =
                crate::graph::storage::disk::type_index::TypeIndexStore::from_base(base);
            loaded = true;
        }
        if !loaded {
            let ti_path = dir.join("type_indices.bin.zst");
            if ti_path.exists() {
                if let Ok(compressed) = std::fs::read(&ti_path) {
                    if let Ok(bytes) = zstd::decode_all(compressed.as_slice()) {
                        match read_type_indices_bin(&bytes, &graph.interner) {
                            Ok(Some(indices)) => {
                                graph.type_indices.replace_with(indices);
                                loaded = true;
                            }
                            _ => {
                                if let Ok(indices) = bincode::deserialize(&bytes) {
                                    graph.type_indices.replace_with(indices);
                                    loaded = true;
                                }
                            }
                        }
                    }
                }
            }
        }
        if !loaded {
            // Fallback: rebuild from node_slots scan
            let mut new_type_indices: std::collections::HashMap<
                String,
                Vec<petgraph::graph::NodeIndex>,
            > = std::collections::HashMap::new();
            for i in 0..dg.node_slots.len() {
                let slot = dg.node_slots.get(i);
                if slot.is_alive() {
                    let key = crate::graph::schema::InternedKey::from_u64(slot.node_type);
                    if let Some(type_name) = graph.interner.try_resolve(key) {
                        new_type_indices
                            .entry(type_name.to_string())
                            .or_default()
                            .push(petgraph::graph::NodeIndex::new(i));
                    }
                }
            }
            graph.type_indices.replace_with(new_type_indices);
        }
    }
    log_stage("type_indices_load", t);

    // Build type_schemas from node_type_metadata (needed for column loading)
    for (node_type, props) in &graph.node_type_metadata {
        let mut schema = crate::graph::schema::TypeSchema::new();
        for prop_name in props.keys() {
            let key = graph.interner.get_or_intern(prop_name);
            schema.add_key(key);
        }
        graph
            .type_schemas
            .insert(node_type.clone(), std::sync::Arc::new(schema));
    }

    // Load column stores — prefer mmap-backed (columns.bin + columns_meta).
    // 0.8.12 phase-1: PR1 phase 4 moved these files to `seg_000/`. Check
    // both locations so post-phase-4 saves still take the fast mmap path
    // — without this the load fell through to the legacy
    // `columns/<type>/columns.zst` branch which returns an empty
    // `column_stores` map, breaking `MATCH (n:Type)` queries after a
    // disk-mode save + reload.
    let mmap_path = {
        let seg0 = dir.join("seg_000/columns.bin");
        if seg0.exists() {
            seg0
        } else {
            dir.join("columns.bin")
        }
    };
    let meta_bin_path = {
        let seg0 = dir.join("seg_000/columns_meta.bin.zst");
        if seg0.exists() {
            seg0
        } else {
            dir.join("columns_meta.bin.zst")
        }
    };
    let meta_json_path = {
        let seg0 = dir.join("seg_000/columns_meta.json");
        if seg0.exists() {
            seg0
        } else {
            dir.join("columns_meta.json")
        }
    };
    let has_mmap = mmap_path.exists() && (meta_bin_path.exists() || meta_json_path.exists());
    let t = stage_timer();
    if has_mmap {
        use crate::graph::io::ntriples::ColumnTypeMeta;
        use memmap2::MmapMut;

        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&mmap_path)?;
        // SAFETY: columns.bin exists in the disk-graph directory and is
        // opened read-write by this loader. KGLite holds the Python GIL
        // during load; no other process writes to the file concurrently.
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        let mmap_arc = std::sync::Arc::new(mmap);

        // Prefer bincode (fast) over JSON (slow for 295 MB)
        let type_metas: Vec<ColumnTypeMeta> = if meta_bin_path.exists() {
            let compressed = std::fs::read(&meta_bin_path)?;
            let bytes = zstd::decode_all(compressed.as_slice()).map_err(io::Error::other)?;
            bincode::deserialize(&bytes).map_err(io::Error::other)?
        } else {
            let meta_json = std::fs::read_to_string(&meta_json_path)?;
            serde_json::from_str(&meta_json).map_err(io::Error::other)?
        };

        for tm in type_metas {
            let store = tm.to_mmap_store(std::sync::Arc::clone(&mmap_arc));
            let cs = crate::graph::storage::column_store::ColumnStore::from_mmap_store(
                std::sync::Arc::new(store),
            );
            graph.column_stores.insert(tm.type_name, Arc::new(cs));
        }

        // Additively load sidecars for types added post-`load_ntriples`
        // via `add_nodes`. The sidecar writer in `DirGraph::save_disk`
        // emits `columns/<type>/columns.zst` only for types NOT in
        // `columns_meta`, so the two paths don't clash — but we still
        // check before overwriting out of caution.
        load_column_sidecars(dir, &mut graph)?;
    } else {
        // Legacy path: load from columns/<type>/columns.zst files
        load_column_sidecars(dir, &mut graph)?;
    }
    log_stage("column_stores_load", t);

    // Sync column stores to DiskGraph
    graph.sync_disk_column_stores();

    // Load id_indices from disk.
    //
    // Three formats, in priority order:
    //   1. id_indices.bin   — 0.8.28+ raw mmap-resident layout (lazy reads,
    //      ~ms load even at Wikidata scale).
    //   2. id_indices.bin.zst with KGLIIDX1 magic — 0.8.13 flat-CSR format
    //      (eager decompress + HashMap rebuild; legacy fallback).
    //   3. id_indices.bin.zst as bincode HashMap — pre-0.8.13 (oldest).
    let t = stage_timer();
    if crate::graph::storage::GraphRead::is_disk(&graph.graph) {
        if let Some(base) =
            crate::graph::storage::disk::id_index::IdIndexBase::load_from(dir, &graph.interner)?
        {
            graph.id_indices = crate::graph::storage::disk::id_index::IdIndexStore::from_base(base);
        } else {
            let id_indices_path = dir.join("id_indices.bin.zst");
            if id_indices_path.exists() {
                if let Ok(compressed) = std::fs::read(&id_indices_path) {
                    if let Ok(bytes) = zstd::decode_all(compressed.as_slice()) {
                        match read_id_indices_bin(&bytes, &graph.interner) {
                            Ok(Some(indices)) => graph.id_indices.replace_with(indices),
                            _ => {
                                if let Ok(indices) = bincode::deserialize(&bytes) {
                                    graph.id_indices.replace_with(indices);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    log_stage("id_indices_load", t);

    // 0.8.28+: `type_connectivity_cache` is populated lazily on first
    // access (in `introspection/describe.rs`'s
    // `compute_type_connectivity` fallback). Pre-loading it eagerly was
    // costing 15+ s on slice-built graphs (128 M triples × 3 String
    // allocations each) for data that most query workloads never touch.
    // Read sites that miss the cache already degrade gracefully to a
    // bounded edge scan.
    //
    // Opt-in eager load: `KGLITE_EAGER_TYPE_CONNECTIVITY=1`. Users that
    // call `describe()` immediately after load can set this to amortize
    // the cost into load instead of the first describe().
    let t = stage_timer();
    if std::env::var_os("KGLITE_EAGER_TYPE_CONNECTIVITY").is_some()
        && !graph.has_type_connectivity_cache()
    {
        if let Ok(Some(triples)) = read_type_connectivity_bin(dir, &graph) {
            if !triples.is_empty() {
                *graph.type_connectivity_cache.write().unwrap() = Some(triples);
            }
        }
    }
    log_stage("type_connectivity_load", t);

    // Load embeddings if present
    let emb_path = dir.join("embeddings.bin.zst");
    if emb_path.exists() {
        if let Ok(compressed) = std::fs::read(&emb_path) {
            if let Ok(bytes) = zstd::decode_all(compressed.as_slice()) {
                if let Ok(embeddings) =
                    bincode::deserialize::<HashMap<(String, String), EmbeddingStore>>(&bytes)
                {
                    graph.embeddings = embeddings;
                }
            }
        }
    }

    // Load timeseries if present
    let ts_path = dir.join("timeseries.bin.zst");
    if ts_path.exists() {
        if let Ok(compressed) = std::fs::read(&ts_path) {
            if let Ok(bytes) = zstd::decode_all(compressed.as_slice()) {
                if let Ok(ts_store) = bincode::deserialize::<HashMap<usize, NodeTimeseries>>(&bytes)
                {
                    graph.timeseries_store = ts_store;
                }
            }
        }
    }

    // Load secondary labels sidecar if present (0.10.5+). Disk's
    // columnar layout has no slot for NodeData.extra_labels, so the
    // sidecar carries the inverted index. Older disk graphs (0.10.4
    // and earlier) won't have this file — that's the graceful single-
    // label degrade path.
    let _ = read_secondary_labels_bin(dir, &mut graph);

    // Backfill the connection_types O(1)-lookup cache from the loaded
    // metadata. The v3 / file loader does this at line 1606 of read_v3;
    // the disk loader was the only path that left it empty and relied
    // on `has_connection_type`'s metadata-fallback branch. The fallback
    // is correct on a freshly-loaded graph but flips into the wrong
    // branch the moment any code path calls `register_connection_type`
    // (which inserts into the cache and trips the "use cache" fast
    // path on subsequent lookups). Backfilling here keeps the cache
    // authoritative throughout the lifetime of the loaded graph.
    graph.build_connection_types_cache();

    log_stage("load_disk_dir_total", _load_t);

    Ok(Arc::new(graph))
}

/// Load `columns/<type>/columns.zst` sidecars into `graph.column_stores`.
/// Skips entries whose type is already loaded (from `columns.bin`'s mmap
/// fast path). Used by both the legacy flat layout and the additive
/// post-`columns.bin` path that covers types added post-build via
/// `add_nodes`.
fn load_column_sidecars(
    dir: &std::path::Path,
    graph: &mut crate::graph::dir_graph::DirGraph,
) -> io::Result<()> {
    use rayon::prelude::*;

    let columns_dir = dir.join("columns");
    if !columns_dir.exists() {
        return Ok(());
    }

    // Collect job descriptors so the heavy work (read + zstd decode +
    // ColumnStore::load_packed) can run in a rayon thread pool. On a
    // 17M-node Wikidata article-author carve with ~4,500 distinct
    // types, the previous sequential loop spent ~70 s in zstd alone;
    // parallelising drops it to a few seconds on a 16-core machine.
    struct Job {
        type_name: String,
        col_file: std::path::PathBuf,
        schema: Arc<crate::graph::schema::TypeSchema>,
        type_meta: std::collections::HashMap<String, String>,
        // Pre-fetched fallback row-count for legacy (pre-0.8.12) sidecars.
        legacy_row_count: u32,
    }

    let mut jobs: Vec<Job> = Vec::new();
    for entry in std::fs::read_dir(&columns_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let type_name = entry.file_name().to_string_lossy().to_string();
        if graph.column_stores.contains_key(&type_name) {
            // columns.bin mmap path already loaded this type.
            continue;
        }
        let col_file = entry.path().join("columns.zst");
        if !col_file.exists() {
            continue;
        }
        let schema = graph
            .type_schemas
            .get(&type_name)
            .cloned()
            .unwrap_or_else(|| std::sync::Arc::new(crate::graph::schema::TypeSchema::new()));
        let type_meta = graph
            .node_type_metadata
            .get(&type_name)
            .cloned()
            .unwrap_or_default();
        let legacy_row_count = graph
            .type_indices
            .get(&type_name)
            .map(|v| v.len() as u32)
            .unwrap_or(0);
        jobs.push(Job {
            type_name,
            col_file,
            schema,
            type_meta,
            legacy_row_count,
        });
    }

    // Decompress + load_packed each sidecar in parallel.
    let interner = &graph.interner;
    let results: Vec<io::Result<(String, crate::graph::storage::column_store::ColumnStore)>> = jobs
        .into_par_iter()
        .map(
            |job| -> io::Result<(String, crate::graph::storage::column_store::ColumnStore)> {
                let compressed = std::fs::read(&job.col_file)?;
                let decoded = zstd::decode_all(compressed.as_slice()).map_err(io::Error::other)?;
                // New format: `KGLCOLv1` + row_count: u32 + packed bytes.
                // Old format (pre-0.8.12): raw packed bytes. Dispatch via the
                // magic tag. Old-format row_count is derived from
                // `type_indices[type].len()` — wrong after DELETE tombstones
                // (0.8.12 CHANGELOG F2), but best effort for legacy graphs.
                let (packed_slice, row_count): (&[u8], u32) =
                    if decoded.len() >= 12 && &decoded[..8] == b"KGLCOLv1" {
                        let rc = u32::from_le_bytes(decoded[8..12].try_into().unwrap());
                        (&decoded[12..], rc)
                    } else {
                        (decoded.as_slice(), job.legacy_row_count)
                    };
                let store = crate::graph::storage::column_store::ColumnStore::load_packed(
                    job.schema,
                    &job.type_meta,
                    interner,
                    packed_slice,
                    row_count,
                    None,
                )?;
                Ok((job.type_name, store))
            },
        )
        .collect();

    for r in results {
        let (type_name, store) = r?;
        graph.column_stores.insert(type_name, Arc::new(store));
    }
    Ok(())
}

/// Load v4 columnar format (Phase A.1 / C5+).
///
/// Same on-disk layout as v3 by section structure; v4 differs by
/// magic bytes + Value enum gaining Node/Relationship/Path/List/Map
/// variants (serde discriminants 9..=13). Old v3 files are rejected
/// at the magic check before they reach this function.
fn load_v4(buf: &[u8]) -> io::Result<Arc<DirGraph>> {
    if buf.len() < 12 {
        return Err(io::Error::other(
            "v4 file is truncated — header incomplete.",
        ));
    }

    // Parse header
    let core_version = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
    let metadata_len = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;

    if core_version > CURRENT_CORE_DATA_VERSION {
        return Err(io::Error::other(format!(
            "File uses core data version {} but this library only supports up to version {}. \
             Please upgrade kglite.",
            core_version, CURRENT_CORE_DATA_VERSION,
        )));
    }

    let metadata_end = 12 + metadata_len;
    if buf.len() < metadata_end {
        return Err(io::Error::other(
            "v4 file is truncated — metadata incomplete.",
        ));
    }

    // Parse JSON metadata
    let metadata: FileMetadata = serde_json::from_slice(&buf[12..metadata_end])
        .map_err(|e| io::Error::other(format!("Failed to parse v3 metadata: {}", e)))?;

    // Section offsets
    let topology_start = metadata_end;
    let topology_end = topology_start + metadata.topology_compressed_size as usize;

    // Decompress + deserialize topology (properties are empty maps)
    let topology_compressed = &buf[topology_start..topology_end];
    let topology_raw = zstd_decompress(topology_compressed)?;

    let mut interner = StringInterner::new();
    let graph: crate::graph::schema::GraphBackend = {
        let _guard = SerdeDeserializeGuard::new(&mut interner);
        bincode_deser(&topology_raw)?
    };
    drop(topology_raw);

    // Extract v3 section metadata before apply_to consumes the rest
    let column_sections = metadata.column_sections.clone();
    let embeddings_compressed_size = metadata.embeddings_compressed_size;
    let timeseries_compressed_size = metadata.timeseries_compressed_size;
    let secondary_labels_compressed_size = metadata.secondary_labels_compressed_size;

    // Reassemble DirGraph
    let mut dir_graph = DirGraph::from_graph(graph);
    dir_graph.interner = interner;
    metadata.apply_to(&mut dir_graph);

    // Rebuild type indices and schemas (needed for ColumnStore construction).
    // Note: rebuild_indices_from_keys is deferred until after column loading
    // because properties are empty at this point (stripped during save).
    dir_graph.rebuild_type_indices_and_compact();
    dir_graph.build_connection_types_cache();

    // Load column sections one type at a time
    let mut section_offset = topology_end;

    // Create temp directory for mmap column files (unique per load to avoid collisions)
    let temp_dir = std::env::temp_dir().join(format!(
        "kglite_v3_{}_{:x}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    // Register for cleanup on DirGraph drop
    if let Ok(mut dirs) = dir_graph.temp_dirs.lock() {
        dirs.push(temp_dir.clone());
    }

    for section_meta in &column_sections {
        let section_end = section_offset + section_meta.compressed_size as usize;
        if buf.len() < section_end {
            return Err(io::Error::other(format!(
                "v3 file truncated — column section '{}' incomplete.",
                section_meta.type_name
            )));
        }

        let compressed = &buf[section_offset..section_end];
        let packed = zstd_decompress(compressed)?;

        // Build schema from the column section metadata (exact match for saved
        // columns). Using type_schemas here would include id/title columns that
        // are NOT in the column data, creating empty placeholder columns that
        // corrupt the file on re-save.
        {
            let col_keys: Vec<crate::graph::schema::InternedKey> = section_meta
                .columns
                .keys()
                .map(|name| {
                    dir_graph.interner.get_or_intern(name);
                    crate::graph::schema::InternedKey::from_str(name)
                })
                .collect();
            let column_schema = Arc::new(crate::graph::schema::TypeSchema::from_keys(col_keys));

            let type_meta = dir_graph
                .node_type_metadata
                .get(&section_meta.type_name)
                .cloned()
                .unwrap_or_default();

            // Create temp dir for this type's column files
            let type_temp_dir = temp_dir.join(&section_meta.type_name);
            std::fs::create_dir_all(&type_temp_dir)?;

            let store = ColumnStore::load_packed(
                column_schema,
                &type_meta,
                &dir_graph.interner,
                &packed,
                section_meta.row_count,
                Some(&type_temp_dir),
            )?;
            drop(packed); // free before next type

            dir_graph
                .column_stores
                .insert(section_meta.type_name.clone(), Arc::new(store));
        }

        section_offset = section_end;
    }

    // Re-point nodes to columnar storage
    for (type_name, store) in &dir_graph.column_stores {
        let has_id_title = store.has_id_title_columns();
        if let Some(indices) = dir_graph.type_indices.get(type_name) {
            for (row_id, node_idx) in indices.iter().enumerate() {
                if let Some(node) = dir_graph.graph.node_weight_mut(node_idx) {
                    node.properties = PropertyStorage::Columnar {
                        store: Arc::clone(store),
                        row_id: row_id as u32,
                    };
                    // Set sentinel values if store has id/title columns (mapped mode)
                    if has_id_title {
                        node.id = Value::Null;
                        node.title = Value::Null;
                    }
                }
            }
        }
    }

    // Now that nodes have columnar properties, rebuild property/range/composite indices
    dir_graph.rebuild_indices_from_keys();

    // Load embeddings if present
    if embeddings_compressed_size > 0 {
        let emb_end = section_offset + embeddings_compressed_size as usize;
        if buf.len() >= emb_end {
            let emb_compressed = &buf[section_offset..emb_end];
            let emb_raw = zstd_decompress(emb_compressed)?;
            let embeddings: HashMap<(String, String), EmbeddingStore> = bincode_deser(&emb_raw)?;
            dir_graph.embeddings = embeddings;
            section_offset = emb_end;
        }
    }

    // Load timeseries if present
    if timeseries_compressed_size > 0 {
        let ts_end = section_offset + timeseries_compressed_size as usize;
        if buf.len() >= ts_end {
            let ts_compressed = &buf[section_offset..ts_end];
            let ts_raw = zstd_decompress(ts_compressed)?;
            let ts_store: HashMap<usize, NodeTimeseries> = bincode_deser(&ts_raw)?;
            dir_graph.timeseries_store = ts_store;
            section_offset = ts_end;
        }
    }

    // Load secondary-label-index section if present (0.10.5+). The
    // interner is fully populated by this point, so InternedKey →
    // String resolution works.
    if secondary_labels_compressed_size > 0 {
        let sl_end = section_offset + secondary_labels_compressed_size as usize;
        if buf.len() >= sl_end {
            let sl_compressed = &buf[section_offset..sl_end];
            let sl_raw = zstd_decompress(sl_compressed)?;
            decode_secondary_label_index(&sl_raw, &mut dir_graph)?;
        }
    }

    Ok(Arc::new(dir_graph))
}

// ─── Embedding Export / Import ────────────────────────────────────────────

use crate::datatypes::values::Value;

/// Magic bytes for the embedding export format.
const KGLE_MAGIC: [u8; 4] = *b"KGLE";
const KGLE_VERSION: u32 = 1;

/// A single embedding store serialized with node IDs (not internal indices).
#[derive(Serialize, Deserialize)]
struct ExportedEmbeddingStore {
    node_type: String,
    text_column: String, // e.g. "summary" (without _emb suffix)
    dimension: usize,
    entries: Vec<(Value, Vec<f32>)>, // (node_id, embedding) pairs
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

        // Resolve node indices → node IDs
        let mut entries: Vec<(Value, Vec<f32>)> = Vec::with_capacity(store.len());
        for &node_index in &store.slot_to_node {
            if let Some(node) = graph
                .graph
                .node_weight(petgraph::graph::NodeIndex::new(node_index))
            {
                if let Some(embedding) = store.get_embedding(node_index) {
                    entries.push((node.id().into_owned(), embedding.to_vec()));
                }
            }
        }

        total_embeddings += entries.len();
        exported_stores.push(ExportedEmbeddingStore {
            node_type: node_type.clone(),
            text_column: text_column.to_string(),
            dimension: store.dimension,
            entries,
        });
    }

    // Write: magic + version + gzip(bincode(stores))
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&KGLE_MAGIC)?;
    writer.write_all(&KGLE_VERSION.to_le_bytes())?;

    let gz = GzEncoder::new(&mut writer, Compression::new(3));
    bincode_options()
        .serialize_into(gz, &exported_stores)
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

    // Decompress and deserialize
    let gz = GzDecoder::new(&buf[8..]);
    let exported_stores: Vec<ExportedEmbeddingStore> = bincode_options()
        .deserialize_from(gz)
        .map_err(|e| io::Error::other(format!("Failed to deserialize embedding data: {}", e)))?;

    let mut total_imported = 0usize;
    let mut total_skipped = 0usize;
    let mut stores_count = 0usize;
    let mut dropped_stores = 0usize;

    for exported in exported_stores {
        // Build ID index for this node type so lookup_by_id works
        graph.build_id_index(&exported.node_type);

        let mut store = crate::graph::schema::EmbeddingStore::new(exported.dimension);
        store
            .data
            .reserve(exported.entries.len() * exported.dimension);

        let mut imported = 0usize;
        let mut skipped = 0usize;

        for (id, vec) in &exported.entries {
            match graph.lookup_by_id(&exported.node_type, id) {
                Some(node_idx) => {
                    store.set_embedding(node_idx.index(), vec);
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
