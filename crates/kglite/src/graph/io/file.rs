// Versioned binary format for KnowledgeGraph persistence.
//
// File format v8 layout (v5, v6 and v7 are identical apart from the magic, the
// per-column encodings and the metadata trailer noted below, and are still read):
//   [0..4]     Magic: b"RGF\x08" (Rusty Graph Format, version 8)
//   [4]        Codec tag: 2 (Postcard v1)
//   [5..9]     core_data_version: u32 LE
//   [9..13]    metadata_length: u32 LE (v8: the JSON plus its 4-byte CRC32)
//   [13..13+N] JSON metadata (column schemas, section sizes, section
//              integrity digests, all config); v8 appends a CRC32 (LE) over
//              bytes [0..13] and the JSON, counted in N
//   [section]  topology.zst — graph structure WITHOUT node properties
//   [section]  columns_<Type>.zst — one per node type, packed column data
//   [section]  embeddings.zst (optional)
//   [section]  edge_embeddings.zst (required for core v4; absent from core v3)
//   [section]  timeseries.zst (optional)
//   [section]  secondary_labels.zst (optional)
//   [section]  vector_index.zst (optional, rebuildable)
//   [section]  text_index.zst (optional, rebuildable)
//   [section]  edge_vector_index.zst (optional, rebuildable; core v4 only)
//   [section]  edge_text_index.zst (optional, rebuildable; ignored by readers
//              that predate it)
//
// v6 vs v5: the section layout, metadata schema and codec are unchanged. What
// v6 adds is a per-column encoding choice inside the packed column sections —
// an `Int64` column may be written as `"int64d"` (zigzag-varint deltas) when
// that is smaller than the fixed-width `"int64"` array, and is re-typed to the
// same in-memory column on load. A v5 reader would take the unknown tag for a
// `Mixed` column and fail decoding it, so the container version is what stops
// it: 0.15.14 refuses a v6 file by version number.
//
// v7 vs v6: the section layout, metadata schema and codec are unchanged. What
// v7 adds is content inside the packed column sections: a `"timestamp"` column
// (microseconds) and an `Int64` title column where v6 wrote them as `Mixed`.
// A v6 reader would misparse those, so it refuses a v7 file by version number
// ("Please upgrade kglite").
//
// v8 vs v7: the section layout and codec are unchanged. v8 adds the metadata
// CRC32 trailer — before it, one flipped byte in the metadata JSON or the length
// fields was parsed (or misparsed) rather than detected — and allows a
// `Value::Duration` with a sub-second part, written under a serde tag a v7
// reader rejects. Section payloads keep their own per-section digests. This
// writer emits v8 only; v5, v6 and v7 are still read, and a v7 reader refuses a
// v8 file by version number (the shipped 0.19.x wording says "older version",
// wrong in direction — see CHANGELOG).
//
// v9 vs v8: v8's bytes, then a data region of raw sections (embedding and
// column payloads) that a reader can map instead of decompress. Written only
// when a save supplies at least one such section, so a graph without them
// produces v8 bytes exactly. Layout:
//   [..]       v8 header, metadata (+ CRC32) and compressed sections, as above
//   [pad]      zeros up to the next 64-byte boundary — the data-region base
//   [region]   mappable sections, each at a 64-byte-aligned offset
// `FileMetadata::mappable` tables them (offset relative to the base, length,
// dtype, flags, CRC32); see `io::file::mappable`. A v8 reader refuses the file
// by container version ("Please upgrade kglite"); `SaveLayout::Compact` is the
// writer's way to keep producing v8.
//
// Pre-v5 magic values are retained only for explicit rejection and migration
// guidance; their payloads are never decoded by the current reader.

#[cfg(test)]
use crate::datatypes::values::Value;
use crate::graph::constraints::{NamedConstraint, UniqueConstraintKey};
use crate::graph::features::temporal::declarations::TemporalDeclarations;
use crate::graph::features::timeseries::{NodeTimeseries, TimeseriesConfig};
use crate::graph::property_types::DeclaredType;
use crate::graph::schema::{
    CompositeIndexKey, ConnectionTypeInfo, ConnectivityTriple, DirGraph, EmbeddingStore, IndexKey,
    PropertyStorage, SaveMetadata, SchemaDefinition, SerdeDeserializeGuard, SerdeSerializeGuard,
    SpatialConfig, StringInterner, StripPropertiesGuard, TemporalConfig,
};
use crate::graph::storage::column_store::ColumnStore;
use crate::graph::storage::mode::StorageMode;
use crate::graph::storage::property_storage::ColumnarRow;
use crate::graph::storage::{GraphRead, GraphWrite};
// Loaders return `Arc<DirGraph>`; each binding wraps that in its own type
// (pyapi → `KnowledgeGraph`, mcp-server → `ActiveGraph`), which keeps io
// decoupled from binding state.
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::graph::io::columns_meta;
use crate::graph::io::magic::{
    newer_portable_format_error, unrecognized_magic_error, MAX_READABLE_CONTAINER_VERSION,
    V3_HARD_BREAK_MSG, V3_MAGIC, V4_MAGIC, V5_MAGIC, V6_MAGIC, V7_MAGIC, V8_MAGIC, V9_MAGIC,
};
use crate::serde_codec;

const MAX_CODEC_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const DISK_SERDE_MAGIC: &[u8; 8] = b"KGLDSC1\0";

/// Highest core data version this reader accepts. Writers select v3 for
/// node-only graphs and v4 when relationship embedding declarations are
/// present, because an older reader must reject rather than silently discard
/// the required relationship-vector section.
///
/// 0.9.52: bumped to 2 — the `Value` enum gained five structured
/// variants (Node, Relationship, Path, List, Map).
///
/// 0.10.29: bumped to 3 — `EmbeddingStore` gained `model_id` +
/// `text_hashes` (positional Serde fields), so an embeddings section
/// written by core-version ≤ 2 can't be deserialized by this binary.
/// Files *without* embeddings load unchanged; a ≤ 2 file *with*
/// embeddings is rejected with a rebuild-and-re-embed message (see
/// `EMBED_FORMAT_BREAK_MSG`). Embeddings are a rebuildable cache, so this
/// is a deliberate, contained break — not a whole-graph format break.
const NODE_ONLY_CORE_DATA_VERSION: u32 = 3;
const CURRENT_CORE_DATA_VERSION: u32 = 4;

/// The first core-data version whose embeddings section carries the
/// `model_id` + `text_hashes` fields. A file below this with a non-empty
/// embeddings section can't be read by this binary.
const EMBED_PROVENANCE_MIN_VERSION: u32 = 3;

// ─── Section integrity ───────────────────────────────────────────────────────
//
// Every section in the container is covered twice:
//
//  1. Its zstd frame carries an XXH64 content checksum (`include_checksum`),
//     which the decoder verifies on its own with no cooperation from us. Costs
//     4 bytes per section and catches damage inside the compressed payload.
//  2. `FileMetadata::section_digests` records a CRC32 of the *compressed*
//     bytes, verified in `SectionCursor::take` before a byte reaches zstd.
//
// Both are cheap enough to run eagerly on every load, but only because the
// CRC is hardware-accelerated: measured on a 180 MB `.kgl` (release, arm64,
// min of 5), layer 1 costs ~20 ms (+4.7%) and layer 2 ~14 ms over a
// digest-free 423 ms load. The software-table CRC that 0.16.6 shipped cost
// ~360 ms (+85%) on its own, which is the regression this note now guards
// against: keep `section_digest` on an accelerated implementation.
//
// Layer 2 exists because layer 1 only protects what a decoder chooses to
// decode: a flipped bit that lands where the section *boundaries* are read
// from, or in a section a reader skips, never reaches an XXH64 check. It also
// covers files written by a build that had checksums off.
//
// Both layers are additive: an older binary ignores the unknown JSON key and
// zstd frames with a content checksum are ordinary zstd frames, so a file
// written here still loads on a build that predates this. A file written
// *before* this carries no digests, and `take` then verifies nothing — exactly
// its previous behaviour.

// Canonical `section_digests` keys for the fixed sections.
const TOPOLOGY_SECTION: &str = "topology";
const EMBEDDINGS_SECTION: &str = "embeddings";
const EDGE_EMBEDDINGS_SECTION: &str = "edge_embeddings";
const TIMESERIES_SECTION: &str = "timeseries";
const SECONDARY_LABELS_SECTION: &str = "secondary_labels";
const VECTOR_INDEX_SECTION: &str = "vector_index";
const TEXT_INDEX_SECTION: &str = "text_index";

/// Canonical `section_digests` key for one node type's column section.
///
/// Namespaced so a node type literally named `embeddings` cannot collide with
/// the fixed keys above, and keyed by *type name* rather than by position so a
/// type appearing, disappearing or sorting differently only moves its own
/// entry.
fn column_section_key(type_name: &str) -> String {
    format!("columns:{type_name}")
}

/// CRC32 (IEEE) of one section's compressed bytes. Shares the WAL's
/// implementation so both integrity checks in this crate agree on what a
/// digest of the same bytes is; see [`crate::graph::wal::crc32`] for why that
/// implementation is hardware-accelerated rather than a software table.
fn section_digest(compressed: &[u8]) -> u32 {
    crate::graph::wal::crc32(compressed)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PortableColumnSection {
    type_name: String,
    compressed_size: u64,
    row_count: u32,
    columns: HashMap<String, String>, // prop_name → type_tag
}

/// Metadata serialized as JSON in portable files. Defaulted additions remain
/// readable when older files omit them.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct FileMetadata {
    /// Core data version at save time — must match or be migratable.
    #[serde(default)]
    core_data_version: u32,
    #[serde(default)]
    library_version: String,
    /// Oldest library version that wrote this data (`SaveMetadata`). Absent
    /// whenever it equals `library_version`, so a graph built and saved by one
    /// version writes the bytes it wrote before the field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    oldest_writer: Option<String>,
    #[serde(default)]
    schema_definition: Option<SchemaDefinition>,
    /// Index keys (property / composite / range) rebuilt after load.
    #[serde(default)]
    property_index_keys: Vec<IndexKey>,
    #[serde(default)]
    composite_index_keys: Vec<CompositeIndexKey>,
    #[serde(default)]
    range_index_keys: Vec<IndexKey>,
    /// Relationship property index keys rebuilt after load. Additive: absent
    /// when empty, so a graph with none writes the bytes it always wrote, and an
    /// older binary ignores the key (it then has no relationship index).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    rel_property_index_keys: Vec<IndexKey>,
    /// Declared UNIQUE constraints to reinstall after load.
    ///
    /// Canonical statement of the additive-and-skipped-when-empty posture every
    /// `skip_serializing_if` field here shares: an older file lacking the key
    /// deserializes to "none declared", exactly its original behaviour, and a
    /// graph that declares none writes byte-identical output to one produced
    /// before the field existed — which is what keeps the `test_phase4_parity`
    /// golden digest stable.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    unique_constraint_keys: Vec<UniqueConstraintKey>,
    /// User-supplied constraint names → the declaration each names, so
    /// `DROP CONSTRAINT <name>` survives save/load.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    constraint_names: HashMap<String, NamedConstraint>,
    /// User-chosen index names → the index each names, so `DROP INDEX <name>`
    /// and the named search procedures survive save/load. Same additive,
    /// skipped-when-empty posture as `constraint_names`.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    index_names: HashMap<String, crate::graph::dir_graph::index_names::NamedIndex>,
    /// Which declared unique constraints came from DDL
    /// (`CREATE CONSTRAINT ... IS UNIQUE` / `... IS NODE KEY`) rather than from
    /// a schema, with each property tuple normalised.
    ///
    /// `unique_constraint_keys` above rebuilds the enforcement; this says *who
    /// declared it*, which is what a schema withdrawal consults before deleting
    /// an index (`DirGraph::withdraw_schema_unique`, called from `set_schema`).
    /// A declaration and a schema primary key on the same `(type, property)`
    /// share one index, so without this field the first `define_schema()` after
    /// a reload deleted a `CREATE CONSTRAINT` it never named — the uniqueness
    /// twin of the presence bug the field below fixes.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    ddl_unique_constraints: BTreeSet<UniqueConstraintKey>,
    /// Which `(node_type, property)` presence constraints were declared through
    /// DDL (`CREATE CONSTRAINT ... IS NOT NULL`) rather than through a schema.
    ///
    /// The *enforced* list rides `schema_definition.required_fields` and needs
    /// nothing here; what needs persisting is the **provenance**, because it is
    /// the only thing that tells a later `define_schema()` it may not withdraw
    /// the declaration (`DirGraph::reapply_ddl_not_null`, called from
    /// `set_schema`). Without this field a reload rebuilt the graph with an
    /// empty provenance set — `DirGraph`'s own serde derive is not the `.kgl`
    /// payload, the load path builds a fresh graph and repopulates it from
    /// *this* struct — so the first unrelated `define_schema()` after a reload
    /// silently un-enforced a constraint the user had written in Cypher.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    ddl_not_null_constraints: BTreeSet<(String, String)>,
    /// Declared property-type constraints (`CREATE CONSTRAINT ... IS :: T`), as
    /// `node_type -> property -> type`.
    ///
    /// Unlike the presence half above, this map *is* the enforcement structure
    /// rather than a provenance record for one — nothing else on the graph
    /// remembers the declared type — so without it a reload silently stops
    /// enforcing every type constraint the file was saved with. `DirGraph`'s own
    /// serde derive does not persist it: the load path builds a fresh graph and
    /// repopulates it from this struct (`from_graph` / `apply_to_with`).
    ///
    /// A file that *does* carry one will not load on a build that predates
    /// `ConstraintKind::PropertyType` — the deliberate one-way format posture,
    /// documented in the CHANGELOG.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    ddl_property_type_constraints: BTreeMap<String, BTreeMap<String, DeclaredType>>,
    /// Declared relationship presence constraints, as
    /// `(connection_type, property)`. Unlike its node counterpart this is not a
    /// provenance record beside a schema list — a connection type has no
    /// `required_fields` — so it *is* the declaration, and a reload without it
    /// silently forgets every relationship presence constraint in the file.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    rel_ddl_not_null_constraints: BTreeSet<(String, String)>,
    /// Declared relationship property-type constraints, as
    /// `connection_type -> property -> type`. Same enforcement-structure role
    /// as the two above.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    rel_ddl_property_type_constraints: BTreeMap<String, BTreeMap<String, DeclaredType>>,
    /// Declared relationship UNIQUE constraints, as `(connection_type,
    /// property)`. Additive like the two above: an older binary ignores the key
    /// and opens the file without the enforcement, and a graph that declares
    /// none writes the bytes it always wrote. The owned index is rebuilt at load.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    rel_ddl_unique_constraints: BTreeSet<(String, String)>,
    /// Node type metadata: node_type → { property_name → type_string }
    #[serde(default)]
    node_type_metadata: HashMap<String, HashMap<String, String>>,
    #[serde(default)]
    connection_type_metadata: HashMap<String, ConnectionTypeInfo>,
    /// Original ID field name per node type (for alias resolution)
    #[serde(default)]
    id_field_aliases: FxHashMap<String, String>,
    /// Original title field name per node type (for alias resolution)
    #[serde(default)]
    title_field_aliases: FxHashMap<String, String>,
    /// Auto-vacuum threshold (None = disabled, default Some(0.3))
    #[serde(default = "crate::graph::dir_graph::default_auto_vacuum_threshold")]
    auto_vacuum_threshold: Option<f64>,
    /// The storage mode that wrote this file, in the cross-binding vocabulary
    /// `StorageMode::as_str` owns. Omitted for a memory graph, so the common
    /// save keeps the pre-field bytes. Never read the raw value: the
    /// `storage_mode` submodule owns what a reader may conclude from it, absent
    /// key included.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    storage_mode: Option<String>,
    /// Parent types: child_type → parent_type. Determines which types are
    /// "core" vs "supporting" in describe() output.
    #[serde(default)]
    parent_types: HashMap<String, String>,
    /// The declared semantic layer (`graph/ontology.rs`). Additive: absent
    /// in older files → empty store; an ontology-free graph writes
    /// byte-identical output (the `constraint_names` posture above, which
    /// keeps the `test_phase4_parity` golden digest stable).
    #[serde(
        default,
        skip_serializing_if = "crate::graph::ontology::OntologyStore::is_empty"
    )]
    ontology: crate::graph::ontology::OntologyStore,
    /// Materialized-label bookkeeping — persisted beside the store so a
    /// reloaded graph keeps its Closed/Open states.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    managed_labels: BTreeMap<String, crate::graph::ontology::ManagedLabelState>,
    /// Table-property fidelity metadata (`tables.rs`). Additive,
    /// skip-when-empty (golden-digest posture).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    table_property_meta: BTreeMap<String, crate::graph::tables::TablePropertyMeta>,
    /// Declared structured property shapes. Same posture.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    property_shapes: BTreeMap<String, crate::graph::tables::PropertyShape>,
    /// Graph-level instructions/briefing per channel (rendered at the top of
    /// describe()). Additive — old files default to empty.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    graph_instructions: HashMap<String, String>,
    /// What the blueprint build that produced the graph reported
    /// (`DirGraph::build_info`). Additive and skipped when absent, so a graph
    /// that was not blueprint-built writes the bytes it wrote before this
    /// existed and the golden digests hold.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    build_info: Option<crate::graph::diagnostics::BuildInfo>,
    /// The graph's stored valid-time default (`all` or a `YYYY-MM-DD` date),
    /// absent for the built-in `today`, so a graph that never set one writes
    /// the bytes it wrote before this existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    valid_time_default: Option<String>,
    /// Where this graph was built from, what that directory looked like, and
    /// which dialect it was read with — `DirGraph::{source_root,
    /// source_fingerprint, source_dialect}`, the vault-provenance set
    /// (VAULT.md §12). A later process reads them to decide whether a rebuild
    /// would read anything new, and *how* to read it; nothing in the engine
    /// can re-derive them from the graph's content.
    ///
    /// Additive and skipped when absent (the `unique_constraint_keys` posture
    /// above): a graph that was not built from a directory writes exactly the
    /// bytes it wrote before these fields existed, which is what keeps every
    /// golden digest stable. `source_dialect` is younger than the other two,
    /// so a `.kgl` written by 0.17.8–0.17.10 carries them and not it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source_root: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source_fingerprint: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source_dialect: Option<String>,
    /// The kglite version that built it and the non-dialect knobs it was
    /// read with — `DirGraph::{source_build_version, source_options}`. Same
    /// additive, skip-when-absent posture as the three above, so a graph that
    /// was not built from a directory still writes the bytes it wrote before
    /// these existed and every golden digest holds. Both are younger than the
    /// other three: a `.kgl` written before 0.17.12 carries neither.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source_build_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source_options: Option<String>,
    /// The caller's own data-model revision (see `DirGraph::user_schema_version`),
    /// carried across save/load so a migration runner can tell which of its
    /// ordered scripts a graph has already had applied. Not an engine version:
    /// `core_data_version` above and the `.kgl` magic own the format lifecycle.
    ///
    /// Omitted at the baseline value; older files lack the key and default to 0.
    #[serde(default, skip_serializing_if = "is_zero")]
    user_schema_version: u32,
    /// Highest WAL log-sequence number this checkpoint already contains (see
    /// `DirGraph::checkpoint_lsn`). On a durable reopen, replay skips every
    /// frame at or below it, so a **stale WAL prefix** — one whose frames
    /// predate the checkpoint — cannot be folded back over a newer snapshot and
    /// roll committed properties backwards.
    ///
    /// The gate is anchored in the checkpoint rather than in the log because the
    /// failure is precisely that the log is stale *relative to* the checkpoint;
    /// only the checkpoint is authoritative about how much of the log it
    /// consumed.
    ///
    /// Omitted at the baseline; older files lack the key and default to 0, i.e.
    /// replay everything, the pre-gate behaviour.
    #[serde(default, skip_serializing_if = "is_zero")]
    checkpoint_lsn: u64,
    /// Where the change-data-capture epoch that was running when this file was
    /// written had got to — see
    /// [`CdcHandoff`](crate::graph::cdc::CdcHandoff).
    ///
    /// Purely a diagnostic. The change log is never persisted (a cursor must
    /// not silently address different data), so this cannot resume a stream;
    /// what it does is let the next process's wrong-epoch refusal say *where
    /// the old epoch ended* instead of only that it ended.
    ///
    /// **Stamped by every save, not only by a durable checkpoint.** A save is
    /// the checkpoint-shaped event for this purpose: the file is what the next
    /// process loads, and that process is the one whose consumer arrives
    /// holding a stale cursor. A save made while capture is off carries
    /// forward whatever stamp the graph already had, because "epoch 7 ended at
    /// 412" stays true after capture stops.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cdc_handoff: Option<crate::graph::cdc::CdcHandoff>,
    #[serde(default)]
    spatial_configs: HashMap<String, SpatialConfig>,
    #[serde(default)]
    timeseries_configs: HashMap<String, TimeseriesConfig>,
    /// The validity-interval configs an older build reads: a mirror of the
    /// declarations it would apply correctly, and the only record in a file
    /// written before `temporal_declarations` existed (see
    /// `features/temporal/persist.rs`). Always written, even empty, so a
    /// graph without declarations keeps its bytes.
    #[serde(default)]
    temporal_node_configs: HashMap<String, TemporalConfig>,
    #[serde(default)]
    temporal_edge_configs: HashMap<String, Vec<TemporalConfig>>,
    /// Every validity-interval declaration; read in place of the two legacy
    /// keys when present. Skipped when empty, for the same bytes.
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "crate::graph::features::temporal::persist::lenient_entries"
    )]
    temporal_declarations: Vec<crate::graph::features::temporal::persist::PersistedDeclaration>,
    /// Timeseries data version: 1 = Vec<Vec<i64>> keys (legacy), 2 = NaiveDate keys.
    #[serde(default = "default_ts_data_version")]
    timeseries_data_version: u32,
    #[serde(default)]
    topology_compressed_size: u64,
    /// v9 data-region section table (see `mappable`); empty — and therefore
    /// absent from the JSON — in every container that carries no such section.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    mappable: Vec<mappable::MapEntry>,
    /// Parts this save wrote without a mapped form and why; empty, and so
    /// absent from the JSON, when there are none. A load reports them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    mapped_fallbacks: Vec<crate::graph::dir_graph::MappedFallback>,
    /// Declared quantized scan copies of embedding stores (see
    /// `quantized_sections`); absent from the JSON when no store has one, so
    /// a graph without quantization keeps its bytes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    quantization: Vec<quantized_sections::QuantRecord>,
    /// Column-section metadata (one per node type).
    #[serde(default)]
    column_sections: Vec<PortableColumnSection>,
    /// Compressed size of the embedding section (0 if none).
    #[serde(default)]
    embeddings_compressed_size: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    edge_embeddings_compressed_size: u64,
    /// Compressed size of the timeseries section (0 if none).
    #[serde(default)]
    timeseries_compressed_size: u64,
    /// 0.10.5: compressed size of secondary-label-index section (0 if
    /// none). Persists `DirGraph.secondary_label_index` for in-memory
    /// graphs. Disk graphs use the parallel `secondary_labels.bin.zst`
    /// sidecar. Older `.kgl` files default to 0 (no section to read).
    #[serde(default)]
    secondary_labels_compressed_size: u64,
    /// 0.11.0: compressed size of the HNSW vector-index section (0 if none).
    /// The section payload is self-describing (magic + format version), so a
    /// reader that doesn't recognise it — or sees a newer index format —
    /// silently skips it and the (rebuildable) index is simply absent. Older
    /// `.kgl` files default to 0.
    #[serde(default)]
    vector_index_compressed_size: u64,
    /// 0.16.10: compressed size of the BM25 text-index section (0 if none).
    /// Self-describing like the vector section above — its own magic and its
    /// own payload version — so a reader that does not recognise the payload
    /// skips it and the (rebuildable) index is simply absent.
    ///
    /// Skipped when zero, unlike `vector_index_compressed_size`: this field
    /// arrived after the byte-level golden digests, so a graph with no text
    /// index has to write exactly the bytes it wrote before the field existed
    /// (`unique_constraint_keys` above states the posture in full).
    #[serde(default, skip_serializing_if = "is_zero")]
    text_index_compressed_size: u64,
    /// Relationship HNSW section size (`edge_vector_persistence`); skipped at zero.
    #[serde(default, skip_serializing_if = "is_zero")]
    edge_vector_index_compressed_size: u64,
    /// Relationship BM25 section size (`text_index_persistence`); skipped at
    /// zero, so a graph without one writes the bytes it wrote before.
    #[serde(default, skip_serializing_if = "is_zero")]
    edge_text_index_compressed_size: u64,
    /// CRC32 (IEEE) of each section's **compressed** bytes, keyed by canonical
    /// section name (see [`column_section_key`] and the "Section integrity"
    /// note above).
    ///
    /// Keyed rather than positional so the map survives an optional section
    /// being absent, a node type being added or removed, and any change in
    /// write order: the reader looks up the one section it is about to take,
    /// and a section with no entry is simply not verified.
    ///
    /// Additive in both directions (see the "Section integrity" note above).
    /// Skipped for the disk-mode `metadata.json`, whose sections live in
    /// separate files with their own integrity story.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    section_digests: BTreeMap<String, u32>,
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

/// `skip_serializing_if` predicate for the additive integer keys
/// (`user_schema_version`, `checkpoint_lsn`): omitting them at the baseline
/// keeps saves byte-identical to pre-field ones. Generic over the integer width
/// so a second such key needs no near-duplicate predicate.
fn is_zero<T: Default + PartialEq>(value: &T) -> bool {
    *value == T::default()
}

impl FileMetadata {
    /// Build metadata from a DirGraph, leaving section sizes at zero (the
    /// caller fills them in after compression).
    pub(crate) fn from_graph(graph: &DirGraph) -> Self {
        Self::from_graph_version(graph, NODE_ONLY_CORE_DATA_VERSION)
    }

    fn from_graph_version(graph: &DirGraph, core_data_version: u32) -> Self {
        let (legacy_nodes, legacy_edges) = graph.temporal.legacy_mirror();
        FileMetadata {
            core_data_version,
            library_version: env!("CARGO_PKG_VERSION").to_string(),
            oldest_writer: Some(graph.save_metadata.oldest_writer_for_save())
                .filter(|oldest| oldest != env!("CARGO_PKG_VERSION")),
            schema_definition: graph.schema_definition.clone(),
            property_index_keys: graph.property_index_keys.clone(),
            composite_index_keys: graph.composite_index_keys.clone(),
            range_index_keys: graph.range_index_keys.clone(),
            rel_property_index_keys: graph.rel.index_keys.clone(),
            unique_constraint_keys: graph.unique_constraint_keys.clone(),
            constraint_names: graph.constraint_names.clone(),
            index_names: graph.index_names.clone(),
            ddl_unique_constraints: graph.ddl_unique_constraints.clone(),
            ddl_not_null_constraints: graph.ddl_not_null_constraints.clone(),
            ddl_property_type_constraints: graph.ddl_property_type_constraints.clone(),
            rel_ddl_not_null_constraints: graph.rel_ddl_not_null_constraints.clone(),
            rel_ddl_property_type_constraints: graph.rel_ddl_property_type_constraints.clone(),
            rel_ddl_unique_constraints: graph.rel.unique.clone(),
            node_type_metadata: (*graph.node_type_metadata).clone(),
            connection_type_metadata: (*graph.connection_type_metadata).clone(),
            id_field_aliases: (*graph.id_field_aliases).clone(),
            title_field_aliases: (*graph.title_field_aliases).clone(),
            auto_vacuum_threshold: graph.auto_vacuum_threshold,
            storage_mode: recorded_storage_mode_tag(graph),
            parent_types: (*graph.parent_types).clone(),
            ontology: (*graph.ontology).clone(),
            managed_labels: graph.managed_labels.clone(),
            table_property_meta: graph.table_property_meta.clone(),
            property_shapes: graph.property_shapes.clone(),
            graph_instructions: graph.graph_instructions.clone(),
            build_info: graph.build_info.clone(),
            valid_time_default: Some(graph.stored_valid_time_default)
                .filter(|stored| {
                    *stored != crate::graph::features::temporal::ValidTimeDefault::Today
                })
                .map(|stored| stored.to_string()),
            source_root: graph.source_root.clone(),
            source_fingerprint: graph.source_fingerprint,
            source_dialect: graph.source_dialect.clone(),
            source_build_version: graph.source_build_version.clone(),
            source_options: graph.source_options.clone(),
            user_schema_version: graph.user_schema_version,
            checkpoint_lsn: graph.checkpoint_lsn,
            // A live log's position wins; otherwise carry forward what an
            // earlier save recorded, which is still true about that epoch.
            cdc_handoff: crate::graph::cdc::status(graph)
                .map(|status| crate::graph::cdc::CdcHandoff {
                    epoch: status.epoch,
                    last_seq: status.current,
                })
                .or(graph.cdc_handoff),
            spatial_configs: graph.spatial_configs.clone(),
            timeseries_configs: graph.timeseries_configs.clone(),
            temporal_node_configs: legacy_nodes,
            temporal_edge_configs: legacy_edges,
            temporal_declarations: graph.temporal.to_persisted(),
            timeseries_data_version: 2,
            topology_compressed_size: 0,
            mappable: Vec::new(),
            mapped_fallbacks: Vec::new(),
            quantization: quantized_sections::collect_records(graph),
            column_sections: Vec::new(),
            embeddings_compressed_size: 0,
            edge_embeddings_compressed_size: 0,
            timeseries_compressed_size: 0,
            secondary_labels_compressed_size: 0,
            vector_index_compressed_size: 0,
            text_index_compressed_size: 0,
            edge_vector_index_compressed_size: 0,
            edge_text_index_compressed_size: 0,
            section_digests: BTreeMap::new(),
            // Persist edge type counts if cache is warm (no O(E) scan if cold)
            edge_type_counts: if graph.has_edge_type_counts_cache() {
                Some((*graph.get_edge_type_counts()).clone())
            } else {
                None
            },
            // 0.8.13: `DirGraph::save_disk` strips this field from the
            // disk-mode metadata.json and writes
            // `type_connectivity.bin.zst` separately (3.17 M-entry JSON
            // list → packed binary). In-memory .kgl saves keep embedding
            // it here for single-file portability.
            type_connectivity: graph.get_type_connectivity().map(|mut triples| {
                crate::graph::introspection::sort_connectivity_triples(&mut triples);
                triples
            }),
        }
    }

    /// Apply metadata fields to a DirGraph during load. Equivalent to
    /// `apply_to_with(graph, true)` — preserved for the in-memory `.kgl`
    /// load path that doesn't have a separate `type_connectivity.bin.zst`.
    pub(crate) fn apply_to(self, graph: &mut DirGraph) {
        self.apply_to_with(graph, true)
    }

    /// Apply metadata fields with control over the type-connectivity derive
    /// fallback. Disk loaders pass `derive_type_connectivity=false`: the
    /// cartesian-product derive over `connection_type_metadata` clones millions
    /// of String triples on large graphs and dominated load time, and the cache
    /// is filled lazily on the first `describe()` instead (or eagerly from
    /// `type_connectivity.bin.zst` under `KGLITE_EAGER_TYPE_CONNECTIVITY`).
    pub(crate) fn apply_to_with(self, graph: &mut DirGraph, derive_type_connectivity: bool) {
        graph.schema_definition = self.schema_definition;
        graph.property_index_keys = self.property_index_keys;
        graph.composite_index_keys = self.composite_index_keys;
        graph.range_index_keys = self.range_index_keys;
        graph.rel.index_keys = self.rel_property_index_keys;
        graph.unique_constraint_keys = self.unique_constraint_keys;
        graph.constraint_names = self.constraint_names;
        graph.index_names = self.index_names;
        graph.ddl_unique_constraints = self.ddl_unique_constraints;
        graph.ddl_not_null_constraints = self.ddl_not_null_constraints;
        graph.ddl_property_type_constraints = self.ddl_property_type_constraints;
        graph.rel_ddl_not_null_constraints = self.rel_ddl_not_null_constraints;
        graph.rel_ddl_property_type_constraints = self.rel_ddl_property_type_constraints;
        graph.rel.unique = self.rel_ddl_unique_constraints;
        graph.property_shapes = self.property_shapes;
        // Read-compat for files saved before declaration refused these.
        for dropped in graph.drop_reserved_provenance_constraints() {
            eprintln!(
                "kglite: dropped a constraint on load ({dropped}): provenance keys are \
                 engine-owned and cannot be constrained"
            );
        }
        graph.node_type_metadata = Arc::new(self.node_type_metadata);
        graph.connection_type_metadata = Arc::new(self.connection_type_metadata);
        graph.id_field_aliases = Arc::new(self.id_field_aliases);
        graph.title_field_aliases = Arc::new(self.title_field_aliases);
        // Read-compat for files saved while a title field naming the id column
        // was recorded as a title spelling.
        graph.drop_identity_shadowing_title_aliases();
        graph.auto_vacuum_threshold = self.auto_vacuum_threshold;
        graph.parent_types = Arc::new(self.parent_types);
        graph.ontology = Arc::new(self.ontology);
        graph.managed_labels = self.managed_labels;
        graph.table_property_meta = self.table_property_meta;
        graph.rebuild_ontology_closures();
        graph.graph_instructions = self.graph_instructions;
        graph.build_info = self.build_info;
        // An unreadable value (hand-edited file) reads as the built-in default.
        let stored = self
            .valid_time_default
            .and_then(|text| crate::graph::features::temporal::ValidTimeDefault::parse(&text).ok())
            .unwrap_or_default();
        graph.stored_valid_time_default = stored;
        graph.valid_time_default = stored;
        graph.source_root = self.source_root;
        graph.source_fingerprint = self.source_fingerprint;
        graph.source_dialect = self.source_dialect;
        graph.source_build_version = self.source_build_version;
        graph.source_options = self.source_options;
        graph.user_schema_version = self.user_schema_version;
        graph.mapped_origin.fallbacks = Arc::new(self.mapped_fallbacks);
        graph.checkpoint_lsn = self.checkpoint_lsn;
        graph.cdc_handoff = self.cdc_handoff;
        graph.spatial_configs = self.spatial_configs;
        graph.timeseries_configs = self.timeseries_configs;
        graph.temporal = TemporalDeclarations::from_file(
            self.temporal_declarations,
            (self.temporal_node_configs, self.temporal_edge_configs),
        );
        // `format_version` is not persisted. A portable load overwrites this
        // with the container byte it parsed (`load_portable_columnar`); a disk
        // directory has no container, so it keeps this build's constant.
        graph.save_metadata = SaveMetadata {
            format_version: crate::graph::schema::KGL_FORMAT_VERSION,
            oldest_writer: self
                .oldest_writer
                .filter(|oldest| !oldest.is_empty())
                .unwrap_or_else(|| self.library_version.clone()),
            library_version: self.library_version,
        };
        if let Some(counts) = self.edge_type_counts {
            *graph.edge_type_counts_cache.write().unwrap() = Some(std::sync::Arc::new(counts));
        }
        let persisted = self
            .type_connectivity
            .filter(|triples| !connectivity_is_fabricated(triples, graph));
        if let Some(triples) = persisted {
            *graph.type_connectivity_cache.write().unwrap() = Some(triples);
        } else if derive_type_connectivity && !graph.connection_type_metadata.is_empty() {
            // Older graphs persist no triples: derive them from
            // connection_type_metadata (instant, no I/O). Only real counts may
            // be derived from: `edge_type_counts` is persisted solely when its
            // cache was warm at save time, and filling in a 0 for every triple
            // when it was cold is not "unknown" to the reader — it is a cache
            // hit, so `get_or_compute_type_connectivity` serves the zeros and
            // never runs the O(E) count that would produce the true numbers.
            // Leaving the cache cold costs one lazy recount and is honest.
            let edge_counts = graph.edge_type_counts_cache.read().unwrap();
            let Some(edge_counts) = edge_counts.as_ref() else {
                return;
            };
            let mut triples = Vec::new();
            for (conn_type, info) in graph.connection_type_metadata.iter() {
                let count = edge_counts.get(conn_type).copied().unwrap_or(0);
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

/// Do persisted connectivity triples carry the fabricated zeros an older build
/// wrote? Such a build derived triples from `connection_type_metadata` with a
/// count of 0 apiece whenever the save carried no `edge_type_counts`, and a
/// later save of that graph persisted the zeros — so the poison round-trips
/// until a load refuses it. An honestly-computed triple set never holds a zero
/// count for a graph with edges: the counter emits a triple only for an edge it
/// walked. Short-circuits on the first non-zero count.
fn connectivity_is_fabricated(
    triples: &[crate::graph::schema::ConnectivityTriple],
    graph: &DirGraph,
) -> bool {
    !triples.is_empty()
        && graph.graph.edge_count() > 0
        && triples.iter().all(|triple| triple.count == 0)
}

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

// The sidecar codec submodules below are split out of this file for the
// production-source file cap; re-exported here so caller paths stay stable.
mod save_temps;
pub use save_temps::reap_stale_save_temps;
pub(crate) use save_temps::save_temp_prefix;

mod metadata_sidecars;
// What a save records about its own storage mode, and what a load may conclude
// from it (`storage_mode` in the metadata above).
mod storage_mode;
pub(crate) use metadata_sidecars::{
    read_connection_type_metadata_bin, read_node_type_metadata_bin,
    write_connection_type_metadata_bin, write_node_type_metadata_bin,
};
use storage_mode::recorded_storage_mode_tag;

pub(crate) mod embedding_sections;
mod fast_load_sidecars;
mod index_sections;
mod integrity;
pub(crate) mod mappable;
mod quantized_sections;
mod save_sections;
mod topology_sections;
use embedding_sections::NodeStoreRecord;
use fast_load_sidecars::{decode_secondary_label_index, encode_secondary_label_index};
pub(crate) use fast_load_sidecars::{
    read_id_indices_bin, read_interner_bin, read_secondary_labels_bin, read_type_connectivity_bin,
    read_type_indices_bin, write_interner_bin, write_secondary_labels_bin,
    write_type_connectivity_bin,
};
#[cfg(test)]
pub(crate) use fast_load_sidecars::{
    ID_INDICES_MAGIC, ID_INDICES_VERSION, TYPE_INDICES_MAGIC, TYPE_INDICES_VERSION,
};
pub use integrity::{verify_file, verify_graph, IntegrityReport, SectionStatus};
use mappable::LoadSource;
pub use mappable::SaveLayout;

// ─── Save ────────────────────────────────────────────────────────────────────

/// Stamp save metadata and snapshot index keys. Quick, runs with GIL held.
pub fn prepare_save(graph: &mut Arc<DirGraph>) {
    let g = crate::graph::handle::make_dir_graph_mut_preserving_lineage(graph);
    let oldest_writer = g.save_metadata.oldest_writer_for_save();
    g.save_metadata = SaveMetadata::current();
    g.save_metadata.oldest_writer = oldest_writer;
    g.populate_index_keys();
}

/// Compress with zstd level 1 (fastest with good ratio) and an XXH64 frame
/// content checksum — layer 1 of the section-integrity story (see the note near
/// the top of this module). Costs 4 bytes per section, and readers built before
/// the flag was set still decode the frame: it is an ordinary zstd frame.
fn zstd_compress(data: &[u8]) -> io::Result<Vec<u8>> {
    let mut encoder = zstd::Encoder::new(Vec::new(), 1)?;
    encoder.include_checksum(true)?;
    encoder.write_all(data)?;
    encoder.finish()
}

thread_local! {
    /// The `max_load_bytes` ceiling of the load running on this thread: no
    /// single decompressed section may exceed it, since a section is held in
    /// memory whole. The metadata-side estimate cannot bound this, because a
    /// crafted file chooses the sizes the estimate reads.
    static LOAD_CEILING: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

/// Holds [`LOAD_CEILING`] for the length of one load.
struct LoadCeilingGuard(Option<u64>);

impl LoadCeilingGuard {
    fn set(ceiling: Option<u64>) -> Self {
        LoadCeilingGuard(LOAD_CEILING.with(|cell| cell.replace(ceiling)))
    }
}

impl Drop for LoadCeilingGuard {
    fn drop(&mut self) {
        LOAD_CEILING.with(|cell| cell.set(self.0));
    }
}

fn zstd_decompress(data: &[u8]) -> io::Result<Vec<u8>> {
    match LOAD_CEILING.with(|cell| cell.get()) {
        Some(ceiling) if ceiling < MAX_DECOMPRESSED_SECTION_BYTES => {
            zstd_decompress_limited(data, ceiling).map_err(|error| {
                invalid_data(format!(
                    "{error}; the ceiling is LoadOptions::max_load_bytes / {MAX_LOAD_ENV_VAR}, \
                     and no single section may decompress past it"
                ))
            })
        }
        _ => zstd_decompress_limited(data, MAX_DECOMPRESSED_SECTION_BYTES),
    }
}

/// Encode a Serde-backed disk sidecar with an explicit codec selector.
/// The explicit frame prevents payloads from being guessed by content.
pub(crate) fn encode_disk_serde<T: Serialize + ?Sized>(value: &T) -> io::Result<Vec<u8>> {
    let payload = serde_codec::encode_versioned(serde_codec::CURRENT_CODEC, value, MAX_CODEC_BYTES)
        .map_err(io::Error::other)?;
    let mut framed = Vec::with_capacity(DISK_SERDE_MAGIC.len() + 1 + payload.len());
    framed.extend_from_slice(DISK_SERDE_MAGIC);
    framed.push(serde_codec::CURRENT_CODEC.tag());
    framed.extend_from_slice(&payload);
    Ok(framed)
}

pub(crate) fn decode_disk_serde<'de, T: Deserialize<'de>>(
    bytes: &'de [u8],
    allocated_bytes: u64,
) -> io::Result<T> {
    if bytes.starts_with(DISK_SERDE_MAGIC) {
        let codec_tag = *bytes
            .get(DISK_SERDE_MAGIC.len())
            .ok_or_else(|| invalid_data("disk codec frame is truncated"))?;
        let payload = &bytes[DISK_SERDE_MAGIC.len() + 1..];
        return serde_codec::decode_exact_with(
            serde_codec::CodecVersion::from_tag(codec_tag)
                .map_err(|e| invalid_data(format!("disk sidecar has an invalid codec tag: {e}")))?,
            payload,
            allocated_bytes,
            serde_codec::DecodeLimits::new(MAX_CODEC_BYTES, MAX_CODEC_BYTES),
        )
        .map_err(|e| invalid_data(format!("disk sidecar payload is undecodable: {e}")));
    }
    Err(pre_014_bincode_error("unframed disk sidecar"))
}

/// Wrap a sidecar decode failure in an error that names the file and
/// tells the operator what to do. Used by `load_disk_sidecars` for optional
/// sidecars (embeddings / timeseries / secondary labels): a *missing*
/// sidecar is legitimate (older graphs), but a present-and-undecodable
/// one is corruption and must fail the load rather than silently
/// loading a graph with data quietly absent.
fn corrupt_sidecar_error(file_name: &str, cause: &io::Error) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!(
            "disk graph sidecar '{file_name}' exists but is corrupt ({cause}); refusing to \
             load the graph with this data silently missing. Restore '{file_name}' from a \
             backup, rebuild the graph, or delete the file to load without it."
        ),
    )
}

/// The most `zstd_decompress_limited` reserves before it has decoded a byte.
/// The declared size is untrusted: a 16-byte frame can claim 16 GiB, and an
/// all-zero section decodes to far more than its stored size, so the buffer
/// grows with the output past this.
const MAX_UPFRONT_RESERVATION: u64 = 64 * 1024 * 1024;

/// Bytes to reserve before decoding `data`: its declared content size, capped
/// at the load limit and at [`MAX_UPFRONT_RESERVATION`].
fn upfront_reservation(data: &[u8], limit: u64) -> usize {
    zstd::zstd_safe::get_frame_content_size(data)
        .ok()
        .flatten()
        .map_or(0, |n| n.min(limit).min(MAX_UPFRONT_RESERVATION) as usize)
}

fn zstd_decompress_limited(data: &[u8], limit: u64) -> io::Result<Vec<u8>> {
    let decoder = zstd::Decoder::new(std::io::Cursor::new(data))
        .map_err(|e| invalid_data(format!("invalid zstd section: {e}")))?;
    let mut bounded = decoder.take(limit.saturating_add(1));
    // Sections are written with a known content size; reserving it up front
    // spares the doubling reallocations (a transient 2x of the output) that
    // `read_to_end` would otherwise make. The header is untrusted, so the
    // reservation is capped at the load limit and at `MAX_UPFRONT_RESERVATION`.
    let mut decoded = Vec::with_capacity(upfront_reservation(data, limit));
    bounded
        .read_to_end(&mut decoded)
        .map_err(|e| invalid_data(format!("invalid zstd section: {e}")))?;
    if decoded.len() as u64 > limit {
        return Err(invalid_data(format!(
            "decompressed section exceeds the {} byte load limit",
            limit
        )));
    }
    Ok(decoded)
}

fn codec_ser<T: Serialize>(codec: serde_codec::CodecVersion, val: &T) -> io::Result<Vec<u8>> {
    serde_codec::encode_versioned(codec, val, MAX_CODEC_BYTES).map_err(io::Error::other)
}

fn codec_deser<'a, T: Deserialize<'a>>(
    codec: serde_codec::CodecVersion,
    buf: &'a [u8],
    allocated_bytes: u64,
) -> io::Result<T> {
    let envelope = serde_codec::PayloadEnvelope::from_tag(
        codec.tag(),
        buf,
        allocated_bytes,
        serde_codec::DecodeLimits::new(MAX_CODEC_BYTES, MAX_CODEC_BYTES),
    )
    .map_err(|e| invalid_data(format!("binary payload envelope is invalid: {e}")))?;
    let decoded = serde_codec::decode_versioned_exact(envelope);
    decoded.map_err(|e| invalid_data(format!("binary deserialization failed: {e}")))
}

/// Verify every InternedKey in the backend's column-store schemas resolves to a
/// string in `graph.interner`. Catches the class of bug where a writer
/// synthesizes a key via `InternedKey::from_str()` (just hashing) and mutates a
/// ColumnStore without first calling `interner.get_or_intern()` — `save()` would
/// then serialize the unregistered key and `load()` would see "<unknown>"
/// property names, silently corrupting the data.
///
/// Surfaced by the 0.8.39 SET master-path bug; any regression of the same shape,
/// in this or any other write path, now fails the save instead.
fn validate_column_keys_registered(graph: &DirGraph) -> io::Result<()> {
    for (type_name, store) in graph.column_stores_by_name() {
        let schema = store.schema();
        for (_slot, key) in schema.iter() {
            if graph.interner.try_resolve(key).is_none() {
                return Err(invalid_data(format!(
                    "ColumnStore for type '{type_name}' contains unregistered InternedKey {}; \
                     refusing to serialize an unknown property name",
                    key.as_u64()
                )));
            }
        }
    }
    Ok(())
}

/// Atomic, durable counterpart of [`write_kgl`]: serialize to a sibling
/// temp file, fsync it (when `fsync`), then atomically rename it over
/// `path`. A crash at any point leaves either the old file or the new one
/// — never a torn/truncated `.kgl`. The temp name embeds the pid and a
/// per-process counter so two processes saving the same path can't
/// clobber each other's in-flight temp (last *rename* wins, cleanly).
/// Unlike disk-graph directories, a standalone `.kgl` path has no
/// `GraphDirectoryLock`: the atomic rename protects readers from torn files but
/// is not a write-ownership lock, so callers must serialize writers if
/// last-writer-wins is not acceptable.
///
/// `fsync = true` (the default via [`write_kgl`]) flushes the file and
/// its parent directory to disk before returning, so the bytes survive an
/// OS/power crash. `fsync = false` keeps the atomic rename (still no torn
/// file) but skips the durability barrier for speed.
pub fn write_kgl_with(graph: &DirGraph, path: &str, fsync: bool) -> io::Result<()> {
    write_kgl_with_stamp(graph, path, fsync, None).map(|_| ())
}

/// [`write_kgl_with`] recording `checkpoint_lsn` in the file's metadata instead
/// of the graph's own stamp (`None` keeps the graph's). The graph is only read,
/// so a published snapshot another thread holds can be written as it stands.
/// Returns the byte length of the file it published.
pub fn write_kgl_with_stamp(
    graph: &DirGraph,
    path: &str,
    fsync: bool,
    checkpoint_lsn: Option<u64>,
) -> io::Result<u64> {
    write_kgl_with_stamp_layout(graph, path, fsync, checkpoint_lsn, SaveLayout::Auto)
}

/// [`write_kgl_with_stamp`] under an explicit [`SaveLayout`].
pub fn write_kgl_with_stamp_layout(
    graph: &DirGraph,
    path: &str,
    fsync: bool,
    checkpoint_lsn: Option<u64>,
    layout: SaveLayout,
) -> io::Result<u64> {
    let dest = Path::new(path);
    let dir = dest.parent().filter(|p| !p.as_os_str().is_empty());

    // Sibling temp path (same directory → rename is atomic on one fs).
    static SAVE_COUNTER: AtomicU64 = AtomicU64::new(0);
    let nonce = SAVE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp_name = format!("{}{}.{}", save_temp_prefix(dest), std::process::id(), nonce);
    let tmp = match dir {
        Some(d) => d.join(&tmp_name),
        None => Path::new(&tmp_name).to_path_buf(),
    };

    // Scope the writer so the File is closed before the rename.
    let write_result = (|| -> io::Result<u64> {
        let file = File::create(&tmp)?;
        let mut writer = BufWriter::new(file);
        write_kgl_to_layout(graph, &mut writer, checkpoint_lsn, layout)?;
        writer.flush()?;
        let file = writer
            .into_inner()
            .map_err(|e| io::Error::other(e.to_string()))?;
        if fsync {
            crate::graph::durable_io::trace::record(|| "kgl temp sync_all".to_string());
            file.sync_all()?;
        }
        Ok(file.metadata()?.len())
    })();

    // On any write error, remove the temp so a failed save leaves no litter.
    let bytes = match write_result {
        Ok(bytes) => bytes,
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
    };

    crate::graph::durable_io::trace::record(|| "kgl rename".to_string());
    if let Err(e) = std::fs::rename(&tmp, dest) {
        let _ = std::fs::remove_file(&tmp);
        return Err(rename_error_with_mapping_hint(e));
    }

    // fsync the directory so the rename itself is durable (the rename can
    // otherwise be lost on a crash even though the file bytes are synced). A
    // failure is the caller's to see: a durable session truncates its log only
    // after this returns Ok.
    if fsync {
        crate::graph::durable_io::trace::record(|| "kgl sync_dir".to_string());
        crate::graph::durable_io::sync_dir(dir.unwrap_or_else(|| Path::new(".")))?;
    }
    Ok(bytes)
}

/// On Windows a rename over a file that something has mapped fails with
/// `ERROR_USER_MAPPED_FILE` (1224); say what to do about it. Any other error
/// passes through.
fn rename_error_with_mapping_hint(error: io::Error) -> io::Error {
    if cfg!(windows) && error.raw_os_error() == Some(1224) {
        return io::Error::new(
            io::ErrorKind::InvalidInput,
            "source file is memory-mapped by this process; save to another path or reload with embeddings='memory'",
        );
    }
    error
}

/// Serialize, compress, and write the graph to a `.kgl` file, atomically
/// and durably (temp + fsync + rename — see [`write_kgl_with`]). Heavy
/// I/O, safe to run without the GIL.
///
/// The bytes are the v8 container: `V8_MAGIC`, an explicit Postcard codec
/// tag, and `CURRENT_CORE_DATA_VERSION`.
///
/// The graph MUST have columnar storage enabled before calling this function;
/// [`prepare_kgl_write`] is the step that does it.
pub fn write_kgl(graph: &DirGraph, path: &str) -> io::Result<()> {
    write_kgl_with(graph, path, true)
}

/// Digest every section that will be written, keyed the way the loader will
/// look each one up ([`column_section_key`] and the `*_SECTION` constants).
///
/// Takes the already-compressed buffers, because the recorded digest covers
/// the bytes as they sit in the file: the reader can then check a section
/// before handing it to zstd, rather than after trusting it enough to
/// decompress it.
fn build_section_digests(
    topology: &[u8],
    column_meta: &[PortableColumnSection],
    column_data: &[Vec<u8>],
    optional: [(&str, Option<&[u8]>); 8],
) -> BTreeMap<String, u32> {
    let mut digests = BTreeMap::new();
    digests.insert(TOPOLOGY_SECTION.to_string(), section_digest(topology));
    for (meta, data) in column_meta.iter().zip(column_data.iter()) {
        digests.insert(column_section_key(&meta.type_name), section_digest(data));
    }
    for (key, data) in optional {
        if let Some(bytes) = data {
            digests.insert(key.to_string(), section_digest(bytes));
        }
    }
    digests
}

/// Serialize the graph's `.kgl` byte stream (header + topology + column /
/// embedding / timeseries / secondary-label sections) into any writer.
/// Factored out of the file path so the same bytes back the atomic file
/// save, an in-memory `to_bytes()`, and a caller-supplied writer — none of
/// them duplicate the section layout.
pub fn write_kgl_to<W: Write>(graph: &DirGraph, writer: &mut W) -> io::Result<()> {
    write_kgl_to_stamped(graph, writer, None)
}

pub use save_sections::{write_kgl_to_layout, write_kgl_to_stamped};

/// Everything a `.kgl` write needs done to the graph before its bytes are
/// produced: stamp the save metadata ([`prepare_save`]), then run the
/// consolidation pass that reclaims rows deleted nodes left behind, restores
/// ascending row order, and re-derives each column's type from its type's
/// metadata. Row order *is* the file's node binding, so a write that skips
/// this can serialize every row against the wrong node.
///
/// This is the single pre-write step for every `.kgl` producer — the
/// path-writing [`save_inmemory_with_layout`] and the buffer-writing bindings
/// (`KnowledgeGraph.to_bytes`) — so neither can drift from the other. A
/// binding that wants the bytes rather than a file calls this, then
/// [`write_kgl_to`] (releasing its runtime's lock around the write).
pub fn prepare_kgl_write(graph: &mut Arc<DirGraph>) {
    prepare_save(graph);
    let dir = crate::graph::handle::make_dir_graph_mut_preserving_lineage(graph);
    dir.enable_columnar();
}

/// In-memory `.kgl` save composing [`prepare_kgl_write`] + [`write_kgl_with`].
/// Public so non-pyo3 consumers (e.g. `kglite-mcp-server`) can save in-memory
/// graphs without duplicating the dispatch logic from `KnowledgeGraph::save`
/// at `src/graph/pyapi/kg_core.rs`.
///
/// Callers under the GIL should release it around `write_kgl`
/// for parallelism with other Python threads — see `kg_core.rs::save`
/// for the canonical split. Rust-only callers (no GIL) just call
/// this directly.
///
/// `fsync` as in [`write_kgl_with`]; `false` is the bench-only fast path.
/// Callers normally use the mode-aware [`save_graph`] / [`save_graph_with`].
///
/// On Windows, a graph whose embeddings are mapped from `path` itself first
/// copies them to the heap, because the rename that publishes the save cannot
/// replace a mapped file there.
pub fn save_inmemory_with_layout(
    graph: &mut Arc<DirGraph>,
    path: &str,
    fsync: bool,
    layout: SaveLayout,
) -> io::Result<()> {
    prepare_kgl_write(graph);
    embedding_sections::release_mappings_of_destination(graph, path, cfg!(windows));
    write_kgl_with_stamp_layout(graph, path, fsync, None, layout)?;
    stamp_written_container(graph, path)
}

/// Record the container version the file at `path` actually carries: the
/// writer picks v8 or v9 from the layout and the section sizes, so only the
/// written header knows.
fn stamp_written_container(graph: &mut Arc<DirGraph>, path: &str) -> io::Result<()> {
    let mut magic = [0u8; 4];
    File::open(path)
        .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut magic))
        .map_err(|e| io_context("reading the header of", Path::new(path), e))?;
    let version = u32::from(magic[3]);
    if graph.save_metadata.format_version != version {
        crate::graph::handle::make_dir_graph_mut_preserving_lineage(graph)
            .save_metadata
            .format_version = version;
    }
    Ok(())
}

/// Mode-aware durable save: dispatches to `DirGraph::save_disk` for
/// disk-backed graphs, `save_inmemory_with_layout` otherwise. This is THE single
/// save-dispatch — the wheel (`KnowledgeGraph::save`), the MCP server,
/// and the C ABI (`kglite_save_graph`) all route through it so dispatch
/// + durability behaviour can't drift between bindings.
///
/// An explicit save to `path`: it is refused (`SaveError::Io`, naming the
/// holder) when another process holds the graph's writer lease, before the
/// write-ahead guard runs. A lease this process holds is not foreign.
/// [`save_graph_with`] does not run the probe — it also serves the close of a
/// handle that opted out of the lease.
pub fn save_graph(graph: &mut Arc<DirGraph>, path: &str) -> Result<(), SaveError> {
    super::open::refuse_foreign_writer(Path::new(path))
        .map_err(|refusal| SaveError::Io(refusal.error.to_string()))?;
    save_graph_with(graph, path, true)
}

/// Durability-parameterized counterpart of [`save_graph`]. The `fsync`
/// flag is threaded to the in-memory `.kgl` write ([`save_inmemory_with`]);
/// disk-backed graphs persist through `DirGraph::save_disk`, which manages
/// its own durability, so the flag does not apply to them.
///
/// Being the single dispatch, this is also where the *write-ahead* rule is
/// enforced: a save that would strand unreplayed frames in front of the
/// checkpoint it writes is refused before the path is touched
/// ([`save_guard`], and [`SaveError::Refused`] for what a binding does with
/// it). A durable owner's own checkpoint is never refused — its prologue
/// stamps `checkpoint_lsn` first.
pub fn save_graph_with(
    graph: &mut Arc<DirGraph>,
    path: &str,
    fsync: bool,
) -> Result<(), SaveError> {
    save_graph_with_layout(graph, path, fsync, SaveLayout::Auto)
}

/// [`save_graph_with`] under an explicit [`SaveLayout`]. A disk-backed graph
/// persists as a directory and ignores the layout.
pub fn save_graph_with_layout(
    graph: &mut Arc<DirGraph>,
    path: &str,
    fsync: bool,
    layout: SaveLayout,
) -> Result<(), SaveError> {
    save_guard::ensure_target_recovered(graph, path)?;
    if graph.graph.is_disk() {
        let dir = crate::graph::handle::make_dir_graph_mut_preserving_lineage(graph);
        return dir.save_disk(path);
    }
    graph.clear_save_notice();
    save_inmemory_with_layout(graph, path, fsync, layout)
        .map_err(|e| SaveError::Io(e.to_string()))?;
    if let rebase::Rebase::Declined(reason) = rebase::rebase_onto_saved(graph, path) {
        graph.note_write_budget_retained(reason);
    }
    Ok(())
}

/// Convert an in-memory (or mapped) graph to disk-backed storage **and publish
/// it at `path`** — the one-call form of `enable_disk_mode()` + `save(path)`,
/// and the only route that keeps the conversion's scratch files off the system
/// temp directory (`DirGraph::enable_disk_mode_at` materializes inside `path`).
///
/// The live handle is left on the published generation, so the graph is in the
/// same shape a fresh `kglite.open(path)` reads: edges mapped, no mutation
/// overlay, and `save()` writing back to this directory. Bindings surface it as
/// `enable_disk_mode(path=...)`; it lives here so the *guard* and the dispatch
/// stay the same ones every other save runs — the write-ahead rule applies to a
/// materialize exactly as it does to a checkpoint, and refusing before the
/// conversion means a refusal leaves the graph untouched.
pub fn materialize_disk_graph(graph: &mut Arc<DirGraph>, path: &str) -> Result<(), SaveError> {
    save_guard::ensure_target_recovered(graph, path)?;
    // `make_dir_graph_mut` (not the lineage-preserving variant `save_graph_with`
    // uses) because this *is* a mutation of the graph's storage: cached plans
    // and result views keyed on the version must see it, exactly as they do for
    // the pathless conversion.
    let dir = crate::graph::handle::make_dir_graph_mut(graph);
    dir.enable_disk_mode_at(path).map_err(SaveError::Io)
}

// ─── Load ────────────────────────────────────────────────────────────────────

/// Below this size, `std::fs::read()` beats mmap (no mmap syscall overhead).
const FILE_MMAP_THRESHOLD: u64 = 65_536; // 64 KB

const MAX_METADATA_BYTES: usize = 64 * 1024 * 1024;
const MAX_DECOMPRESSED_SECTION_BYTES: u64 = 16 * 1024 * 1024 * 1024;

/// The load path's error-kind rule, in one place.
///
/// **`InvalidData` is a statement about the bytes**: truncation, a magic this
/// build has no reader for, a metadata parse failure, a digest mismatch, a
/// deliberate format break (v3, v4, pre-provenance embeddings), a container or
/// core-data version from the future. Every consumer classifies on it —
/// `kglite-c`'s `classify_io_error` maps it to `KGLITE_ERR_FILE_FORMAT`
/// ("file isn't a valid `.kgl`") and the wheel's `load_err_to_pyerr` to
/// `FileFormatError` — and all of them mean the same thing to the operator:
/// rebuild this file from its source, retrying the read will not help.
///
/// A **syscall** failure keeps whatever kind the OS gave it (`NotFound`,
/// `PermissionDenied`, …) and gains its operation and path through
/// [`io_context`]. `ErrorKind::Other` is left for the residue: a failure that
/// is about neither the bytes nor a syscall, such as a storage-mode conversion
/// that could not be completed after the file decoded fine.
///
/// **`OutOfMemory` is a statement about this process, not about the file**: the
/// load-memory ceiling refused a file that is perfectly valid and would load
/// elsewhere (see [`load_memory_refusal`]). It is a separate kind precisely
/// because `InvalidData`'s advice — "rebuild this file from its source" — is
/// wrong for it, and every binding maps it to its own `LoadMemoryLimit` class
/// rather than to a corrupt-file one.
fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

/// Name the operation and the path on an `io::Error` a syscall produced.
///
/// A load reports through `io::Error`, so an unwrapped one reaching a caller
/// says only that *something* under `load_file` failed: a downstream spent two
/// debugging sessions on a bare `Os { code: 17 }` and a bare
/// `Os { code: 22 }` out of this call (0.16.14), with no way to tell which
/// syscall, which path, or even whether the error was ours.
///
/// The kind is preserved because consumers classify on it, and the OS message
/// keeps its own `(os error N)` suffix, so the only thing the rewrap costs is
/// the `raw_os_error()` accessor — which nothing in the tree, or in any
/// binding, reads off a load.
pub(crate) fn io_context(operation: &str, path: impl AsRef<Path>, error: io::Error) -> io::Error {
    io::Error::new(
        error.kind(),
        format!("{operation} '{}': {error}", path.as_ref().display()),
    )
}

/// Which portable container `prefix` names, or the refusal its magic earns.
///
/// The four entry points that reach a `.kgl` header — the mmap and read halves
/// of [`load_file_with`], [`load_kgl_bytes_with`], and the metadata-head read
/// behind [`estimate_load_memory`] — must agree on every one of these answers,
/// including which deliberate format break gets which message. `origin` names
/// what was refused, since that is the only thing they differ in.
///
/// `prefix` must be at least 4 bytes; each caller checks that first, because
/// "too small to be a `.kgl`" is phrased in terms of what the caller was
/// handed (a file, a buffer).
fn portable_container_format(prefix: &[u8], origin: &str) -> io::Result<&'static str> {
    if prefix[..4] == V9_MAGIC {
        Ok("v9")
    } else if prefix[..4] == V8_MAGIC {
        Ok("v8")
    } else if prefix[..4] == V7_MAGIC {
        Ok("v7")
    } else if prefix[..4] == V6_MAGIC {
        Ok("v6")
    } else if prefix[..4] == V5_MAGIC {
        Ok("v5")
    } else if prefix[..4] == V4_MAGIC {
        Err(pre_014_bincode_error(".kgl container v4"))
    } else if prefix[..4] == V3_MAGIC {
        Err(invalid_data(V3_HARD_BREAK_MSG))
    } else if prefix[..3] == V8_MAGIC[..3] && prefix[3] > MAX_READABLE_CONTAINER_VERSION {
        Err(newer_portable_format_error(prefix[3]))
    } else {
        Err(unrecognized_magic_error(&prefix[..4], origin))
    }
}

pub(crate) fn pre_014_bincode_error(artifact: &str) -> io::Error {
    invalid_data(format!(
        "Unsupported pre-0.14 bincode persistence: {artifact}. This build reads Postcard \
         persistence only. Open the artifact with kglite 0.13.4 and re-save or re-export it, \
         then retry; alternatively rebuild it from the original source."
    ))
}

/// Load the `.kgl` checkpoint (or disk-graph directory) at `path`.
///
/// **The checkpoint only.** Any write-ahead sidecar beside it is neither read
/// nor consulted, deliberately: this is the primitive the durable path is
/// built on — a durable owner loads the checkpoint here and *then* replays the
/// log over it ([`crate::graph::durability::open_log`]) — so a recovery check
/// at this level would make recovery itself unreachable. It is also the way to
/// read a graph another process is writing durably, where a sidecar running
/// ahead of the checkpoint is the steady state rather than a fault.
///
/// A caller that takes the path over — one that may later save back to it —
/// wants [`crate::graph::io::open::open_or_create_graph`], which adds the
/// recovery refusal, or a durable open, which replays.
///
/// Reading a `.kgl` that carries a column blob of 256 KB or more spills it to a
/// temp directory to be mmap'd, with the ownership rules, `KGLITE_TMPDIR`
/// override and orphan sweep [`load_kgl_bytes`] describes. A smaller `.kgl`
/// touches no path but this one.
///
/// Loads with [`LoadOptions::new`]; [`load_file_with`] is the same function
/// taking those options explicitly.
pub fn load_file(path: &str) -> io::Result<Arc<DirGraph>> {
    load_file_with(path, &LoadOptions::new())
}

/// [`load_file`], with the caller's [`LoadOptions`].
///
/// The two options are resolved at opposite ends of the load:
/// `options.storage` below the decode (before a section is decompressed, so an
/// unserveable request is refused cheaply) and `options.defer_index_rebuild` at
/// the very end, where the declared indexes would otherwise be built. Read
/// [`LoadOptions`] before reaching for `storage` as a memory lever — it is not
/// one for a `.kgl`.
pub fn load_file_with(path: &str, options: &LoadOptions) -> io::Result<Arc<DirGraph>> {
    let mut graph = with_load_advisories(load_file_inner(path, options)?)?;
    if let Some(g) = Arc::get_mut(&mut graph) {
        let file_bytes = std::fs::metadata(path).map_or(0, |m| m.len());
        g.enroll_write_budget(options.write_budget, file_bytes);
        g.load_policy = options.policy();
    }
    Ok(graph)
}

/// Record the data advisories of a graph that was just loaded, while it is
/// still uniquely held.
fn with_load_advisories(mut graph: Arc<DirGraph>) -> io::Result<Arc<DirGraph>> {
    if let Some(g) = Arc::get_mut(&mut graph) {
        let t = crate::graph::io::load_timing::stage_timer();
        g.advisories = crate::graph::advisories::compute_advisories(g);
        crate::graph::io::load_timing::log_stage("data_advisories", t);
    }
    Ok(graph)
}

fn load_file_inner(path: &str, options: &LoadOptions) -> io::Result<Arc<DirGraph>> {
    let p = std::path::Path::new(path);
    if p.is_dir() {
        // The directory *is* the graph, so a request for a portable mode has
        // nothing to convert — the same structural refusal
        // `convert_dir_graph_to_mode` gives, raised before the load rather than
        // after it.
        if let Some(requested) = options.storage {
            if requested != StorageMode::Disk {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    crate::graph::storage::mode::disk_conversion_refusal(
                        StorageMode::Disk,
                        requested,
                    ),
                ));
            }
        }
        // Disk identity indexes stay mapped; optional heap index and constraint
        // declarations are always deferred until the first write.
        return load_disk_dir(p);
    }

    let file = File::open(path).map_err(|e| io_context("opening", p, e))?;
    let file_len = file
        .metadata()
        .map_err(|e| io_context("reading the metadata of", p, e))?
        .len();

    // For large files, mmap avoids the full copy into a Vec<u8>. The mapping
    // is an `Arc<MappedFile>` so a section handle can outlive this call.
    if file_len >= FILE_MMAP_THRESHOLD {
        let mapped = mappable::MappedFile::map(file, p)?;
        let bytes = mapped.bytes();
        if bytes.len() < 4 {
            return Err(invalid_data("File is too small to be a valid kglite file."));
        }
        let format_name = portable_container_format(bytes, &format!("'{path}'"))?;
        let source = LoadSource {
            bytes,
            file: Some(&mapped),
        };
        return load_portable_container(&source, format_name, options);
    }

    let buf = std::fs::read(path).map_err(|e| io_context("reading", p, e))?;
    if buf.len() < 4 {
        return Err(invalid_data("File is too small to be a valid kglite file."));
    }
    let format_name = portable_container_format(&buf, &format!("'{path}'"))?;
    let source = LoadSource {
        bytes: &buf,
        file: None,
    };
    load_portable_container(&source, format_name, options)
}

/// Load an in-memory graph from a `.kgl` byte buffer — the counterpart of
/// [`write_kgl_to`] / `KnowledgeGraph.to_bytes()`. Same magic/version
/// validation and error classification as [`load_file`]'s small-file branch.
/// Disk-mode graphs are a directory, not a byte stream, so this only handles
/// the single-file in-memory format.
///
/// **It does not read the `.kgl` from disk — the caller already holds the
/// bytes. It is not filesystem-free** *when it has something big to map*: a
/// column blob of 256 KB or more is written to a per-load spill directory
/// (`$TMPDIR/kglite_portable_<pid>_<tick><seq>/type_<n>/`) so the column can be
/// mmap'd instead of heap-allocated. Smaller columns stay on the heap and no
/// directory is created — a graph whose every column is small touches nothing.
/// The paths of the ones that are created are registered on the returned
/// graph, and the last `DirGraph` holding them removes the tree in `Drop` — so
/// they live exactly as long as the graph.
///
/// **That drop is the only cleanup on this process's side.** A process killed
/// by a signal, an OOM kill or a panic-abort never runs it and leaves the tree
/// behind, and no OS sweep reclaims it on a live macOS login session. What
/// does is the next spilling load in *any* process: it removes sibling
/// directories under the same root whose encoded pid names no running process
/// and whose mtime is over an hour old (unix only — see
/// `file::spill_dirs`). Set `KGLITE_TMPDIR` to put the spills, and that sweep,
/// somewhere other than `$TMPDIR`.
pub fn load_kgl_bytes(data: &[u8]) -> io::Result<Arc<DirGraph>> {
    load_kgl_bytes_with(data, &LoadOptions::new())
}

/// [`load_kgl_bytes`], with the caller's [`LoadOptions`] — the byte-buffer
/// counterpart of [`load_file_with`], resolving both options identically.
pub fn load_kgl_bytes_with(data: &[u8], options: &LoadOptions) -> io::Result<Arc<DirGraph>> {
    if data.len() < 4 {
        return Err(invalid_data(
            "Byte buffer is too small to be a valid kglite graph.",
        ));
    }
    let format_name = portable_container_format(data, "the byte buffer")?;
    let source = LoadSource {
        bytes: data,
        file: None,
    };
    with_load_advisories(load_portable_container(&source, format_name, options)?)
}

/// Contained break message for a pre-v3 embeddings section (model_id +
/// text_hashes added in core-version 3). Only files *with* embeddings hit
/// this; everything else loads. Embeddings are a rebuildable cache.
const EMBED_FORMAT_BREAK_MSG: &str =
    "This .kgl was saved with an older embedding format (before per-vector model \
     id + text-hash provenance, kglite 0.10.29). Its embeddings can't be loaded by \
     this binary. The graph's nodes/edges are fine — reload, re-run \
     embed_texts()/add_embeddings() to rebuild the vectors, and save again. \
     (Embeddings are a rebuildable cache; only the vector section broke.)";

/// Build `type_schemas` from `node_type_metadata`, which column loading needs.
///
/// The catalogue is `Arc`-shared (`dir_graph::schema_cow`), so holding a second
/// handle for the walk costs a refcount and frees `graph` for the
/// `type_schemas_mut()` writes inside it. Nothing in the loop writes the
/// catalogue, so the handle and the field stay the same map throughout.
fn rebuild_disk_type_schemas(graph: &mut DirGraph) -> io::Result<()> {
    let metadata = std::sync::Arc::clone(&graph.node_type_metadata);
    for (node_type, props) in metadata.iter() {
        let mut schema = crate::graph::schema::TypeSchema::new();
        // Sorted: `props` is a `HashMap` and this path has no recorded column
        // order to recover (unlike the portable column sections, whose packed
        // payload carries one). Name order is the canonical choice here — see
        // `TypeSchema` slot-order rule in `dir_graph::rebuild_type_schemas`.
        let mut prop_names: Vec<&String> = props.keys().collect();
        prop_names.sort();
        for prop_name in prop_names {
            let key = graph
                .interner
                .try_get_or_intern(prop_name)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            schema.add_key(key);
        }
        graph
            .type_schemas_mut()
            .insert(node_type.clone(), std::sync::Arc::new(schema));
    }
    Ok(())
}

fn load_disk_dir(dir: &std::path::Path) -> io::Result<Arc<DirGraph>> {
    use crate::graph::io::load_timing::{log_stage, stage_timer};
    use crate::graph::schema::GraphBackend;

    let _load_t = stage_timer();
    let resolved = crate::graph::storage::disk::generation::resolve_snapshot(dir)?;
    let logical_root = resolved.logical_root;
    let snapshot_dir = resolved.snapshot_dir;
    let dir = snapshot_dir.as_path();

    if !dir.join("disk_graph_meta.json").exists() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Directory does not contain a valid disk graph (missing disk_graph_meta.json)",
        ));
    }

    let mut graph = DirGraph::new();

    let t = stage_timer();
    apply_disk_metadata(dir, &mut graph)?;
    log_stage("metadata_json", t);

    // Load interner. Current `interner.bin.zst` carries a codec frame and
    // Postcard `Vec<String>`; unframed binary data is rejected. The older
    // `interner.json` representation remains a read-only data fallback.
    let t = stage_timer();
    let loaded_from_bin = read_interner_bin(dir, &mut graph)?;
    if !loaded_from_bin && dir.join("interner.json").exists() {
        let interner_str = std::fs::read_to_string(dir.join("interner.json"))?;
        let interner_map: std::collections::HashMap<String, String> =
            serde_json::from_str(&interner_str)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        for original in interner_map.values() {
            graph
                .interner
                .try_get_or_intern(original)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        }
    }
    log_stage("interner_load", t);

    // Load DiskGraph — compressed files decompressed to temp dir, then mmap'd.
    // The disk storage loader owns the interner mutably while assembling all
    // stores; current edge-property payloads contain resolved raw key hashes.
    let t = stage_timer();
    let (mut disk_graph, temp_dir) =
        crate::graph::storage::disk::graph::DiskGraph::load_from_dir(dir, &mut graph.interner)?;
    disk_graph.set_logical_root(logical_root);
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
    // This is the `.kgl` → `KnowledgeGraph` construction boundary;
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
    //   3. node_slots scan fallback for graphs missing or pre-0.14 files.
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
                    if let Ok(bytes) = zstd_decompress(&compressed) {
                        if let Ok(Some(indices)) = read_type_indices_bin(&bytes, &graph.interner) {
                            graph.type_indices.replace_with(indices);
                            loaded = true;
                        }
                    }
                }
            }
        }
        if !loaded {
            let mut new_type_indices: std::collections::HashMap<
                String,
                Vec<petgraph::graph::NodeIndex>,
            > = std::collections::HashMap::new();
            for i in 0..dg.node_slot_len() {
                let slot = dg.node_slot(i);
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

    rebuild_disk_type_schemas(&mut graph)?;

    load_disk_column_stores(dir, &mut graph)?;

    legacy_references::normalize_disk_snapshot(&mut graph)?;

    // No sync: the stores were installed straight onto the backend, which is
    // their only owner — there is no DirGraph↔DiskGraph mirror.

    // Load id_indices from disk.
    //
    // Two formats, in priority order:
    //   1. id_indices.bin   — 0.8.28+ raw mmap-resident layout (lazy reads,
    //      ~ms load even at Wikidata scale).
    //   2. id_indices.bin.zst with KGLIIDX1 magic — 0.8.13 flat-CSR format
    //      (eager decompress + HashMap rebuild; retained data fallback).
    // Pre-0.14 bincode caches are ignored and rebuilt lazily.
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
                    if let Ok(bytes) = zstd_decompress(&compressed) {
                        if let Ok(Some(indices)) = read_id_indices_bin(&bytes, &graph.interner) {
                            graph.id_indices.replace_with(indices);
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
            if !triples.is_empty() && !connectivity_is_fabricated(&triples, &graph) {
                *graph.type_connectivity_cache.write().unwrap() = Some(triples);
            }
        }
    }
    log_stage("type_connectivity_load", t);

    load_disk_sidecars(dir, &mut graph)?;

    // Keep declared heap indexes visible without rebuilding large disk graphs
    // at open. The existing write boundary materializes their enforcement.
    graph.defer_index_rebuild_from_keys();

    // Backfill the connection_types O(1)-lookup cache from the loaded metadata.
    // Left empty, `has_connection_type`'s metadata-fallback branch is correct on
    // a freshly-loaded graph but flips into the wrong branch the moment anything
    // calls `register_connection_type` (which inserts into the cache and trips
    // the "use cache" fast path on subsequent lookups).
    graph.build_connection_types_cache();

    log_stage("load_disk_dir_total", _load_t);

    Ok(Arc::new(graph))
}

/// Load a disk graph's embeddings / timeseries / secondary-label sidecars.
/// Split out of `load_disk_dir` to keep it under the function-complexity
/// ceiling; cold load-time path with one shared fail-loud-on-corruption policy.
fn load_disk_sidecars(dir: &std::path::Path, graph: &mut DirGraph) -> io::Result<()> {
    // Absent file = fine (older graphs, or no embeddings); present-but-
    // undecodable is corruption — see [`corrupt_sidecar_error`].
    let emb_path = dir.join("embeddings.bin.zst");
    if emb_path.exists() {
        let mut embeddings = (|| -> io::Result<HashMap<(String, String), EmbeddingStore>> {
            let compressed = std::fs::read(&emb_path)?;
            let bytes = zstd_decompress(&compressed)?;
            decode_disk_serde(&bytes, bytes.capacity() as u64)
                .map_err(|e| invalid_data(e.to_string()))
        })()
        .map_err(|e| corrupt_sidecar_error("embeddings.bin.zst", &e))?;
        // `norms` is `#[serde(skip)]` — validate its source columns, then
        // recompute from `data` post-load.
        validate_and_rebuild_embedding_norms(&mut embeddings)
            .map_err(|e| corrupt_sidecar_error("embeddings.bin.zst", &e))?;
        graph.embeddings = embeddings;
    }

    let edge_emb_path = dir.join("edge_embeddings.bin.zst");
    let edge_embeddings_required =
        crate::graph::storage::disk::graph_persist::edge_embeddings_required(dir)?;
    if edge_embeddings_required {
        if !edge_emb_path.is_file() {
            return Err(corrupt_sidecar_error(
                "edge_embeddings.bin.zst",
                &invalid_data("required relationship-embedding sidecar is missing"),
            ));
        }
        let decoded = (|| -> io::Result<
            BTreeMap<
                crate::graph::edge_embeddings::EdgeEmbeddingKey,
                crate::graph::edge_embeddings::PersistedEdgeEmbeddingStore,
            >,
        > {
            let compressed = std::fs::read(&edge_emb_path)?;
            let bytes = zstd_decompress(&compressed)?;
            decode_disk_serde(&bytes, bytes.capacity() as u64)
                .map_err(|error| invalid_data(error.to_string()))
        })()
        .map_err(|error| corrupt_sidecar_error("edge_embeddings.bin.zst", &error))?;
        let stores =
            crate::graph::edge_embeddings::validate_decoded_edge_embedding_stores(graph, decoded)
                .map_err(|error| {
                corrupt_sidecar_error("edge_embeddings.bin.zst", &invalid_data(error))
            })?;
        graph.edge_embeddings = stores;
    }

    let ts_path = dir.join("timeseries.bin.zst");
    if ts_path.exists() {
        graph.timeseries_store = (|| -> io::Result<HashMap<usize, NodeTimeseries>> {
            let compressed = std::fs::read(&ts_path)?;
            let bytes = zstd_decompress(&compressed)?;
            decode_disk_serde(&bytes, bytes.capacity() as u64)
                .map_err(|e| invalid_data(e.to_string()))
        })()
        .map_err(|e| corrupt_sidecar_error("timeseries.bin.zst", &e))?;
    }

    // Secondary labels (0.10.5+): disk's columnar layout has no slot for
    // NodeData.extra_labels, so the sidecar carries the inverted index. Older
    // disk graphs (0.10.4 and earlier) won't have this file — that's the
    // graceful single-label degrade path (the reader returns Ok(false) when
    // absent).
    read_secondary_labels_bin(dir, graph)
        .map_err(|e| corrupt_sidecar_error("secondary_labels.bin.zst", &e))?;
    Ok(())
}

/// Read a disk graph's `metadata.json` and apply it to `graph`. A directory
/// without the file is legitimate (nothing to apply).
///
/// The two heavy HashMap fields (`node_type_metadata`,
/// `connection_type_metadata`) come from dedicated binary sidecars (0.8.28+)
/// when present — they cost 4-5 s of JSON parse on slice-built Wikidata graphs
/// with 30K-50K types, vs <100 ms in the binary form. Older graphs keep the
/// fields embedded in metadata.json and are picked up by the JSON parse here.
fn apply_disk_metadata(dir: &std::path::Path, graph: &mut DirGraph) -> io::Result<()> {
    if !dir.join("metadata.json").exists() {
        return Ok(());
    }
    let meta_bytes = std::fs::read(dir.join("metadata.json"))?;
    let mut meta: FileMetadata = serde_json::from_slice(&meta_bytes)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    // The directory is the graph, so the mode is not in doubt — but a recorded
    // value that disagrees with it, or one this build cannot recognise, is
    // corruption and must fail here rather than be ignored.
    meta.validate_disk_storage_mode()?;
    if let Some(ntm) = read_node_type_metadata_bin(dir)? {
        meta.node_type_metadata = ntm;
    }
    if let Some(ctm) = read_connection_type_metadata_bin(dir)? {
        meta.connection_type_metadata = ctm;
    }
    // `false`: skip the load-time cartesian-product derive of
    // `type_connectivity` — see [`FileMetadata::apply_to_with`].
    meta.apply_to_with(graph, false);
    Ok(())
}

/// Decode a v5, v6, v7, v8 or v9 container. They share a header, a codec and a section
/// layout; they differ only in which per-column encodings the column sections
/// may use, and the column reader dispatches on the section's own type tags.
/// `format_name` is the version the caller matched, and appears in errors.
fn load_portable_container(
    source: &LoadSource<'_>,
    format_name: &str,
    options: &LoadOptions,
) -> io::Result<Arc<DirGraph>> {
    let buf = source.bytes;
    if buf.len() < 13 {
        return Err(invalid_data(format!(
            "{format_name} file is truncated — header incomplete"
        )));
    }
    let codec = serde_codec::CodecVersion::from_tag(buf[4]).map_err(|e| {
        invalid_data(format!(
            "{format_name} header has an invalid codec tag: {e}"
        ))
    })?;
    if codec != serde_codec::CodecVersion::PostcardV1 {
        return Err(invalid_data(format!(
            "{format_name} header selects codec {}, but {format_name} requires Postcard codec {}",
            codec.tag(),
            serde_codec::CodecVersion::PostcardV1.tag()
        )));
    }
    let core_version = u32::from_le_bytes([buf[5], buf[6], buf[7], buf[8]]);
    let metadata_len = u32::from_le_bytes([buf[9], buf[10], buf[11], buf[12]]) as usize;
    load_portable_columnar(
        source,
        format_name,
        PortableHeader {
            codec,
            core_version,
            metadata_len,
            metadata_start: 13,
        },
        options,
    )
}

struct PortableSectionPlan {
    columns: Vec<PortableColumnSection>,
    embeddings: u64,
    edge_embeddings: u64,
    timeseries: u64,
    secondary_labels: u64,
    vector_index: u64,
    text_index: u64,
    edge_vector_index: u64,
    edge_text_index: u64,
    quantization: Vec<quantized_sections::QuantRecord>,
}

fn parse_portable_metadata<'a>(
    buf: &'a [u8],
    format_name: &str,
    metadata_len: usize,
    metadata_start: usize,
) -> io::Result<(FileMetadata, SectionCursor<'a>)> {
    let metadata_end = metadata_start
        .checked_add(metadata_len)
        .ok_or_else(|| invalid_data(format!("{format_name} metadata offset overflow")))?;
    let metadata_bytes = buf.get(metadata_start..metadata_end).ok_or_else(|| {
        invalid_data(format!(
            "{format_name} file is truncated — metadata incomplete"
        ))
    })?;
    let metadata = decode_metadata_json(&buf[..metadata_start], metadata_bytes, format_name)?;
    let digests = metadata.section_digests.clone();
    Ok((metadata, SectionCursor::new(buf, metadata_end, digests)?))
}

/// Parse the metadata JSON block and apply the guard every reader of it needs.
///
/// Shared by the loader and by the head-only read behind
/// [`estimate_load_memory`] so the two cannot disagree about which metadata is
/// acceptable — an estimate that parsed a block the load then rejects would put
/// a number on a file that never opens.
///
/// `header` is the container's 13 fixed bytes and `metadata_block` the
/// `metadata_length` bytes after them. A v8 block ends in a CRC32 over the
/// header and the JSON; it is verified before the JSON is parsed, so a flipped
/// byte in the metadata or in the length fields is refused as corruption rather
/// than misread.
fn decode_metadata_json(
    header: &[u8],
    metadata_block: &[u8],
    format_name: &str,
) -> io::Result<FileMetadata> {
    let metadata_bytes = if matches!(format_name, "v8" | "v9") {
        let split = metadata_block.len().checked_sub(4).ok_or_else(|| {
            invalid_data(format!(
                "{format_name} metadata is too short to hold its checksum — the file is corrupt"
            ))
        })?;
        let (json, trailer) = metadata_block.split_at(split);
        let stored = u32::from_le_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]);
        let actual = metadata_checksum(header, json);
        if stored != actual {
            return Err(invalid_data(format!(
                "{format_name} header/metadata checksum mismatch (stored {stored:#010x}, \
                 computed {actual:#010x}) — the file is corrupt. Restore it from a backup or \
                 rebuild the graph from its source."
            )));
        }
        json
    } else {
        metadata_block
    };
    let metadata: FileMetadata = serde_json::from_slice(metadata_bytes)
        .map_err(|e| invalid_data(format!("failed to parse {format_name} metadata: {e}")))?;
    if metadata.column_sections.len() > 1_000_000 {
        return Err(invalid_data(format!(
            "{format_name} metadata declares too many column sections"
        )));
    }
    Ok(metadata)
}

/// Absolute offset of a v9 container's data region: the end of the compressed
/// sections, rounded up to the section alignment. Refuses a container whose
/// declared compressed sections run past `file_len`.
fn data_region_base(
    metadata: &FileMetadata,
    metadata_end: usize,
    file_len: u64,
) -> io::Result<u64> {
    let sizes = [
        metadata.topology_compressed_size,
        metadata
            .column_sections
            .iter()
            .fold(0u64, |a, c| a.saturating_add(c.compressed_size)),
        metadata.embeddings_compressed_size,
        metadata.edge_embeddings_compressed_size,
        metadata.timeseries_compressed_size,
        metadata.secondary_labels_compressed_size,
        metadata.vector_index_compressed_size,
        metadata.text_index_compressed_size,
        metadata.edge_vector_index_compressed_size,
        metadata.edge_text_index_compressed_size,
    ];
    let compressed_end = sizes
        .iter()
        .fold(metadata_end as u64, |a, n| a.saturating_add(*n));
    if compressed_end > file_len {
        return Err(invalid_data(
            "file is truncated — the compressed sections run past the end of the file",
        ));
    }
    Ok(mappable::align_up(
        compressed_end,
        mappable::SECTION_ALIGNMENT,
    ))
}

/// Validate a v9 container's data-region table before any section is decoded,
/// refuse a table in a container that cannot carry one, and open its sections
/// (mapped or copied per `options`).
///
/// A v5-v8 file with a `mappable` key was not written by this code: v8 readers
/// never look at the key, so honouring it would read offsets nobody aligned.
fn open_mappable_table(
    metadata: &FileMetadata,
    format_name: &str,
    source: &LoadSource<'_>,
    metadata_end: usize,
    options: &LoadOptions,
    into_mapped: bool,
) -> io::Result<mappable::MappableTable> {
    if metadata.mappable.is_empty() {
        return Ok(mappable::MappableTable::default());
    }
    if format_name != "v9" {
        return Err(invalid_data(format!(
            "{format_name} container declares mappable sections, which only v9 containers carry"
        )));
    }
    let base = data_region_base(metadata, metadata_end, source.bytes.len() as u64)?;
    embedding_sections::open_table(
        &metadata.mappable,
        source.bytes,
        base,
        source.file,
        options,
        into_mapped,
    )
}

/// CRC32 over a v8 container's fixed 13-byte header and its metadata JSON —
/// the magic, codec tag, core-data version, the metadata length and every
/// metadata byte. Stored little-endian as the last 4 bytes of the metadata
/// block.
pub(crate) fn metadata_checksum(header: &[u8], json: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(header);
    hasher.update(json);
    hasher.finalize()
}

/// Bytes in the largest unit that keeps them readable.
///
/// Unit-picking rather than fixed MB: the ceiling is as legitimate on a 40 KB
/// graph as on a 40 GB one, and rounding a 280-byte term to "1 MB" would put a
/// false number in a message whose whole job is to be believed.
fn human_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{} MB", bytes / MB)
    } else if bytes >= KB {
        format!("{} KB", bytes / KB)
    } else {
        format!("{bytes} B")
    }
}

/// The refusal [`LoadOptions::max_load_bytes`] produces: what was estimated,
/// what the ceiling is, which terms make up the estimate, and the two ways out.
///
/// `ErrorKind::OutOfMemory`, not `InvalidData` — see the error-kind rule on
/// [`invalid_data`]. The file is fine; this process's budget is not, and a
/// binding must not tell the operator to rebuild a graph that is not broken.
fn load_memory_refusal(
    estimate: &load_estimate::LoadMemoryEstimate,
    limit: u64,
    options: &LoadOptions,
) -> io::Error {
    let projected = estimate.projected_peak_bytes(options.defer_index_rebuild);
    let index_term_is_payable = !options.defer_index_rebuild && estimate.index_rebuild_bytes > 0;
    let mut message = format!(
        "loading this .kgl is estimated to peak at {} of memory, over the {} ceiling this load \
         was given (LoadOptions::max_load_bytes / {MAX_LOAD_ENV_VAR}). Nothing was decompressed. \
         Estimated terms: {} for the graph's {} node rows and their columns",
        human_bytes(projected),
        human_bytes(limit),
        human_bytes(estimate.section_heap_bytes),
        estimate.node_rows,
    );
    if index_term_is_payable {
        message.push_str(&format!(
            ", {} to rebuild the {} declared index(es)",
            human_bytes(estimate.index_rebuild_bytes),
            estimate.declared_indexes,
        ));
    }
    message.push_str(&format!(
        ", and {} held transiently while the largest section decompresses. ",
        human_bytes(estimate.transient_peak_bytes),
    ));
    if index_term_is_payable {
        message.push_str(&format!(
            "Two ways forward: raise the ceiling, or load with defer_index_rebuild, which records \
             the file's declared indexes instead of building them — that removes the {} index \
             term (measured −42.8% settled footprint on a 500k-row four-index fixture) at the \
             price of a scan where an indexed lookup would have run, and a one-time build on the \
             first write. ",
            human_bytes(estimate.index_rebuild_bytes),
        ));
    } else {
        message.push_str(
            "The index rebuild is already deferred, so the way forward is to raise the ceiling — \
             what remains is the graph itself. ",
        );
    }
    message.push_str(
        "This is an ESTIMATE read from the file's metadata head, not a measurement: it ran \
         0.56×–1.30× of measured settled footprint across the calibration fixtures and errs high \
         on the peak, so it can refuse a load that would have fitted.",
    );
    io::Error::new(io::ErrorKind::OutOfMemory, message)
}

/// The fields [`load_portable_container`] reads out of a v5-v8 container's
/// fixed-size header, carried as one value so the loader's signature does not
/// grow a parameter per header field.
struct PortableHeader {
    codec: serde_codec::CodecVersion,
    core_version: u32,
    metadata_len: usize,
    /// Byte offset of the metadata JSON — immediately after the header.
    metadata_start: usize,
}

/// Load the shared v5/v6/v7 columnar section layout through the codec selected by
/// the already-validated container header.
fn load_portable_columnar(
    source: &LoadSource<'_>,
    format_name: &str,
    header: PortableHeader,
    options: &LoadOptions,
) -> io::Result<Arc<DirGraph>> {
    let buf = source.bytes;
    let PortableHeader {
        codec,
        core_version,
        metadata_len,
        metadata_start,
    } = header;
    if metadata_len > MAX_METADATA_BYTES {
        return Err(invalid_data(format!(
            "{format_name} metadata is {metadata_len} bytes; limit is {MAX_METADATA_BYTES}"
        )));
    }

    if core_version > CURRENT_CORE_DATA_VERSION {
        return Err(invalid_data(format!(
            "File uses core data version {} but this library only supports up to version {}. \
             Please upgrade kglite.",
            core_version, CURRENT_CORE_DATA_VERSION,
        )));
    }
    let _ceiling = LoadCeilingGuard::set(options.max_load_bytes);
    let (metadata, mut sections) =
        parse_portable_metadata(buf, format_name, metadata_len, metadata_start)?;
    if metadata.core_data_version != core_version {
        return Err(invalid_data(
            "portable header and metadata core-data versions disagree",
        ));
    }
    // Whether the graph will be placed in mapped storage decides how the index
    // and topology sections are opened; `topology="memory"` opens them as a
    // memory-storage load does, unread. A mode that does not parse is refused
    // further down.
    let into_mapped = match options.storage {
        Some(requested) => requested == StorageMode::Mapped,
        None => metadata.portable_storage_mode().ok() == Some(StorageMode::Mapped),
    } && options.topology != MapPolicy::Memory;
    let mappable_table = open_mappable_table(
        &metadata,
        format_name,
        source,
        metadata_start + metadata_len,
        options,
        into_mapped,
    )?;
    match core_version {
        CURRENT_CORE_DATA_VERSION if metadata.edge_embeddings_compressed_size == 0 => {
            return Err(invalid_data(
                "core data version 4 requires a non-empty edge_embeddings section",
            ));
        }
        0..=NODE_ONLY_CORE_DATA_VERSION if metadata.edge_embeddings_compressed_size != 0 => {
            return Err(invalid_data(
                "edge_embeddings section requires core data version 4",
            ));
        }
        _ => {}
    }
    // The ceiling is checked here, on the metadata side of the decode: a load
    // refused for memory must not first allocate the memory. See
    // `LoadOptions::max_load_bytes`.
    let normalization_budget = if let Some(limit) = options.max_load_bytes {
        let estimate = load_estimate::estimate_from_metadata(&metadata);
        let base_estimated_peak = estimate.projected_peak_bytes(options.defer_index_rebuild);
        if base_estimated_peak > limit {
            return Err(load_memory_refusal(&estimate, limit, options));
        }
        Some(legacy_references::NormalizationBudget {
            base_estimated_peak,
            limit,
        })
    } else {
        None
    };
    // Resolved before a section is decompressed, so an unplaceable mode fails
    // before the expensive part.
    let recorded_mode = metadata.portable_storage_mode()?;
    // The caller's request outranks the file's record; the record is the
    // fallback. Resolved here, on the same side of the decode, so a request
    // this build cannot serve costs the metadata read and nothing else.
    let target_mode = match options.storage {
        // No `.kgl` can become a directory. Refused with the same wording
        // `convert_dir_graph_to_mode` gives after a load, minus the load.
        Some(StorageMode::Disk) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                crate::graph::storage::mode::disk_conversion_refusal(
                    recorded_mode,
                    StorageMode::Disk,
                ),
            ))
        }
        Some(requested) => requested,
        None => recorded_mode,
    };
    let mut indexes =
        into_mapped.then(|| index_sections::IndexSections::from_table(&mappable_table));
    // A mapped load serves the graph from the file's raw topology when it
    // recorded one, instead of decoding the compressed copy into the heap.
    let frozen = into_mapped
        .then(|| topology_sections::TopologySections::from_table(&mappable_table))
        .flatten();
    let (mut dir_graph, plan) =
        decode_portable_topology(codec, &mut sections, metadata, indexes.as_mut(), frozen)?;
    // The container version is the file's own header byte, not a constant of
    // this build: since v9 a save writes v8 or v9 depending on its layout.
    dir_graph.save_metadata.format_version = u32::from(buf[3]);
    column_sections::load_portable_columns(
        codec,
        &mut dir_graph,
        &mut sections,
        &plan.columns,
        &mappable_table,
        buf,
    )?;
    // Keep declarations visible to complete-state validation while derived
    // lookup structures are deliberately absent. Normalization must precede
    // every index that would otherwise retain raw NodeRef equality.
    dir_graph.defer_index_rebuild_from_keys();
    let normalization_effects =
        legacy_references::normalize_complete_snapshot(&mut dir_graph, normalization_budget)?;
    if !options.defer_index_rebuild {
        dir_graph.materialize_indexes();
    }
    load_portable_optional_sections(
        codec,
        core_version,
        &mut dir_graph,
        &mut sections,
        &plan,
        &normalization_effects,
        &mappable_table,
    )?;
    // Place the graph in its mode. The compressed payload always deserializes
    // into a memory backend (`GraphBackend`'s Deserialize), so a mapped target
    // is swapped onto the mapped backend here — after the graph is complete, and
    // by moving the topology rather than copying it. Memory (and a file that
    // recorded nothing, with nothing requested) is already what the decode
    // produced, so it is a no-op. A mapped load that found the file's raw
    // topology sections never decoded the payload: its backend is already the
    // frozen mapped one, which is a no-op here too.
    crate::graph::storage::mode::convert_dir_graph_to_mode(&mut dir_graph, target_mode)
        .map_err(io::Error::other)?;
    if let Some(indexes) = indexes.filter(|_| dir_graph.graph.is_mapped()) {
        let base = indexes.peer_base(&dir_graph)?;
        dir_graph.graph.install_peer_base(base);
    }
    if options.verify == Verify::Eager {
        if let Some(first) = structure_checks::check_structure(&dir_graph)
            .into_iter()
            .next()
        {
            return Err(invalid_data(format!("{}: {}", first.section, first.error)));
        }
    }
    if let Some(file) = source.file {
        sections.release_consumed(file);
    }
    let origin = &mut dir_graph.mapped_origin;
    origin.was_frozen = dir_graph.graph.is_frozen();
    origin.topology_reason = (!origin.was_frozen).then(|| {
        crate::graph::dir_graph::mapped_report::not_served_reason(
            target_mode == StorageMode::Mapped,
            options.topology == MapPolicy::Memory,
            &origin.fallbacks,
        )
    });
    Ok(Arc::new(dir_graph))
}

mod column_sections;
mod columns;
mod section_cursor;
use section_cursor::SectionCursor;
pub(crate) mod rebase;
pub(crate) use columns::remap_column_stores_to_generation;
use columns::{attach_portable_column_stores, load_disk_column_stores};

mod legacy_references;
mod metadata_head;
mod structure_checks;
pub(crate) use metadata_head::{
    checkpoint_lsn_from_file, read_metadata_head, read_metadata_head_from_file,
};

mod load_estimate;
pub use load_estimate::{estimate_load_memory, estimate_load_memory_bytes, LoadMemoryEstimate};

mod load_options;
pub(crate) use load_options::LoadPolicy;
use load_options::MAX_LOAD_ENV_VAR;
pub use load_options::{LoadOptions, MapPolicy, Verify};

mod portable_sections;
use portable_sections::{
    compressed_len, decode_portable_topology, encode_portable_edge_embeddings,
    load_portable_optional_sections, validate_and_rebuild_embedding_norms, write_optional_section,
};
mod edge_vector_persistence;
use edge_vector_persistence::{encode_edge_vector_index_section, EDGE_VECTOR_INDEX_SECTION};
mod spill_dirs;
pub(crate) use spill_dirs::memory_limit_temp_dir;
use spill_dirs::portable_temp_dir;

mod save_guard;
pub use save_guard::SaveError;

mod text_index_persistence;
use text_index_persistence::{
    decode_edge_text_indexes, decode_text_indexes_after_normalization,
    encode_edge_text_index_section, encode_text_indexes, EDGE_TEXT_INDEX_SECTION,
};

#[cfg(test)]
#[path = "file/container_test_support.rs"]
pub(crate) mod container_test_support;

#[cfg(test)]
#[path = "file/edge_embedding_persistence_tests.rs"]
mod edge_embedding_persistence_tests;

#[cfg(test)]
#[path = "file/mappable_tests.rs"]
mod mappable_tests;

#[cfg(test)]
#[path = "file/column_mapped_tests.rs"]
mod column_mapped_tests;
#[cfg(test)]
mod crafted_structure_tests;
#[cfg(test)]
mod crafted_topology_tests;
#[cfg(test)]
mod eligibility_tests;
#[cfg(test)]
#[path = "file/embedding_load_validation_tests.rs"]
mod embedding_load_validation_tests;
#[cfg(test)]
#[path = "file/embedding_mapped_tests.rs"]
mod embedding_mapped_tests;
#[cfg(test)]
mod frozen_overlay_tests;
#[cfg(test)]
mod frozen_tests;
#[cfg(test)]
#[path = "file/index_sections_tests.rs"]
mod index_sections_tests;
#[cfg(test)]
mod load_release_tests;
#[cfg(test)]
mod mapped_report_tests;
#[cfg(test)]
mod mutation_tests;
#[cfg(test)]
mod rebase_policy_tests;
#[cfg(test)]
mod rollback_overlay_tests;
#[cfg(test)]
mod write_budget_tests;

/// Save `graph` and load it back into mapped storage with the topology sections
/// the floors would otherwise withhold from a small graph, so a fixture can run
/// on a frozen backend. The file is gone before this returns; a Unix mapping
/// keeps its inode, and elsewhere the sections are copied at load.
#[cfg(test)]
pub(crate) fn reload_frozen_for_test(graph: DirGraph) -> DirGraph {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("frozen-fixture.kgl");
    let mut shared = Arc::new(graph);
    index_sections::with_floors(1, 1, || {
        save_graph_with_layout(
            &mut shared,
            path.to_str().expect("utf-8 path"),
            true,
            SaveLayout::Auto,
        )
    })
    .expect("save fixture");
    let loaded = load_file_with(
        path.to_str().expect("utf-8 path"),
        &LoadOptions::new().with_storage(StorageMode::Mapped),
    )
    .expect("load fixture");
    drop(tmp);
    match Arc::try_unwrap(loaded) {
        Ok(graph) => graph,
        Err(_) => panic!("a fresh load is uniquely owned"),
    }
}
#[cfg(test)]
mod quantized_persistence_tests;

mod vector_persistence;

use vector_persistence::{decode_vector_indexes, encode_vector_indexes};
pub use vector_persistence::{
    export_embeddings_to_file, import_embeddings_from_file, EmbeddingExportFilter, ExportStats,
    ImportStats,
};
#[cfg(test)]
#[path = "file_deferred_index_tests.rs"]
mod file_deferred_index_tests;
#[cfg(test)]
#[path = "file_load_ceiling_tests.rs"]
mod file_load_ceiling_tests;
#[cfg(test)]
#[path = "file_load_contract_tests.rs"]
mod file_load_contract_tests;
#[cfg(test)]
#[path = "file_load_options_tests.rs"]
mod file_load_options_tests;
#[cfg(test)]
#[path = "file_tests.rs"]
mod file_tests;
