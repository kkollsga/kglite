// Versioned binary format for KnowledgeGraph persistence.
//
// File format v6 layout (v5 is identical apart from the magic and the
// per-column encodings noted below, and is still read):
//   [0..4]     Magic: b"RGF\x06" (Rusty Graph Format, version 6)
//   [4]        Codec tag: 2 (Postcard v1)
//   [5..9]     core_data_version: u32 LE
//   [9..13]    metadata_length: u32 LE
//   [13..13+N] JSON metadata (column schemas, section sizes, section
//              integrity digests, all config)
//   [section]  topology.zst — graph structure WITHOUT node properties
//   [section]  columns_<Type>.zst — one per node type, packed column data
//   [section]  embeddings.zst (optional)
//   [section]  timeseries.zst (optional)
//   [section]  secondary_labels.zst (optional)
//   [section]  vector_index.zst (optional, rebuildable)
//
// v6 vs v5: the section layout, metadata schema and codec are unchanged. What
// v6 adds is a per-column encoding choice inside the packed column sections —
// an `Int64` column may be written as `"int64d"` (zigzag-varint deltas) when
// that is smaller than the fixed-width `"int64"` array, and is re-typed to the
// same in-memory column on load. A v5 reader would take the unknown tag for a
// `Mixed` column and fail decoding it, so the container version is what stops
// it: this writer emits v6 only, and 0.15.14 refuses it by version number.
//
// Pre-v5 magic values are retained only for explicit rejection and migration
// guidance; their payloads are never decoded by the current reader.

use crate::datatypes::values::Value;
use crate::graph::constraints::{NamedConstraint, UniqueConstraintKey};
use crate::graph::features::timeseries::{NodeTimeseries, TimeseriesConfig};
use crate::graph::property_types::DeclaredType;
use crate::graph::schema::{
    CompositeIndexKey, ConnectionTypeInfo, ConnectivityTriple, DirGraph, EmbeddingStore, IndexKey,
    PropertyStorage, SaveMetadata, SchemaDefinition, SerdeDeserializeGuard, SerdeSerializeGuard,
    SpatialConfig, StringInterner, StripPropertiesGuard, TemporalConfig,
};
use crate::graph::storage::column_store::ColumnStore;
use crate::graph::storage::property_storage::ColumnarRow;
use crate::graph::storage::{GraphRead, GraphWrite};
// Loaders return `Arc<DirGraph>`; each binding wraps that in its own type
// (pyapi → `KnowledgeGraph`, mcp-server → `ActiveGraph`), which keeps io
// decoupled from binding state.
use memmap2::Mmap;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use crate::graph::io::magic::{
    newer_portable_format_error, unrecognized_magic_error, V3_HARD_BREAK_MSG, V3_MAGIC, V4_MAGIC,
    V5_MAGIC, V6_MAGIC,
};
use crate::serde_codec;

const MAX_CODEC_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const DISK_SERDE_MAGIC: &[u8; 8] = b"KGLDSC1\0";

/// Current core data version. Bump ONLY when NodeData, EdgeData, or Value enum changes.
/// This is independent of metadata — metadata uses JSON and handles changes via serde defaults.
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
const CURRENT_CORE_DATA_VERSION: u32 = 3;

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
    #[serde(default)]
    schema_definition: Option<SchemaDefinition>,
    /// Index keys (property / composite / range) rebuilt after load.
    #[serde(default)]
    property_index_keys: Vec<IndexKey>,
    #[serde(default)]
    composite_index_keys: Vec<CompositeIndexKey>,
    #[serde(default)]
    range_index_keys: Vec<IndexKey>,
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
    /// Temporal configuration per node type (valid_from/valid_to on nodes).
    #[serde(default)]
    temporal_node_configs: HashMap<String, TemporalConfig>,
    /// Temporal configuration per connection type (valid_from/valid_to on edges).
    #[serde(default)]
    temporal_edge_configs: HashMap<String, Vec<TemporalConfig>>,
    /// Timeseries data version: 1 = Vec<Vec<i64>> keys (legacy), 2 = NaiveDate keys.
    #[serde(default = "default_ts_data_version")]
    timeseries_data_version: u32,
    #[serde(default)]
    topology_compressed_size: u64,
    /// Column-section metadata (one per node type).
    #[serde(default)]
    column_sections: Vec<PortableColumnSection>,
    /// Compressed size of the embedding section (0 if none).
    #[serde(default)]
    embeddings_compressed_size: u64,
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
        FileMetadata {
            core_data_version: CURRENT_CORE_DATA_VERSION,
            library_version: env!("CARGO_PKG_VERSION").to_string(),
            schema_definition: graph.schema_definition.clone(),
            property_index_keys: graph.property_index_keys.clone(),
            composite_index_keys: graph.composite_index_keys.clone(),
            range_index_keys: graph.range_index_keys.clone(),
            unique_constraint_keys: graph.unique_constraint_keys.clone(),
            constraint_names: graph.constraint_names.clone(),
            ddl_unique_constraints: graph.ddl_unique_constraints.clone(),
            ddl_not_null_constraints: graph.ddl_not_null_constraints.clone(),
            ddl_property_type_constraints: graph.ddl_property_type_constraints.clone(),
            rel_ddl_not_null_constraints: graph.rel_ddl_not_null_constraints.clone(),
            rel_ddl_property_type_constraints: graph.rel_ddl_property_type_constraints.clone(),
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
            temporal_node_configs: graph.temporal_node_configs.clone(),
            temporal_edge_configs: graph.temporal_edge_configs.clone(),
            timeseries_data_version: 2,
            topology_compressed_size: 0,
            column_sections: Vec::new(),
            embeddings_compressed_size: 0,
            timeseries_compressed_size: 0,
            secondary_labels_compressed_size: 0,
            vector_index_compressed_size: 0,
            text_index_compressed_size: 0,
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
        graph.unique_constraint_keys = self.unique_constraint_keys;
        graph.constraint_names = self.constraint_names;
        graph.ddl_unique_constraints = self.ddl_unique_constraints;
        graph.ddl_not_null_constraints = self.ddl_not_null_constraints;
        graph.ddl_property_type_constraints = self.ddl_property_type_constraints;
        graph.rel_ddl_not_null_constraints = self.rel_ddl_not_null_constraints;
        graph.rel_ddl_property_type_constraints = self.rel_ddl_property_type_constraints;
        graph.node_type_metadata = Arc::new(self.node_type_metadata);
        graph.connection_type_metadata = Arc::new(self.connection_type_metadata);
        graph.id_field_aliases = Arc::new(self.id_field_aliases);
        graph.title_field_aliases = Arc::new(self.title_field_aliases);
        graph.auto_vacuum_threshold = self.auto_vacuum_threshold;
        graph.parent_types = Arc::new(self.parent_types);
        graph.ontology = Arc::new(self.ontology);
        graph.managed_labels = self.managed_labels;
        graph.table_property_meta = self.table_property_meta;
        graph.property_shapes = self.property_shapes;
        graph.rebuild_ontology_closures();
        graph.graph_instructions = self.graph_instructions;
        graph.user_schema_version = self.user_schema_version;
        graph.checkpoint_lsn = self.checkpoint_lsn;
        graph.cdc_handoff = self.cdc_handoff;
        graph.spatial_configs = self.spatial_configs;
        graph.timeseries_configs = self.timeseries_configs;
        graph.temporal_node_configs = self.temporal_node_configs;
        graph.temporal_edge_configs = self.temporal_edge_configs;
        // `format_version` is not persisted, so the load side has to re-derive
        // it: the constant this build writes, not a literal pinned to whatever
        // container was current when this line was written.
        graph.save_metadata = SaveMetadata {
            format_version: crate::graph::schema::KGL_FORMAT_VERSION,
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
mod metadata_sidecars;
// What a save records about its own storage mode, and what a load may conclude
// from it (`storage_mode` in the metadata above).
mod storage_mode;
pub(crate) use metadata_sidecars::{
    read_connection_type_metadata_bin, read_node_type_metadata_bin,
    write_connection_type_metadata_bin, write_node_type_metadata_bin,
};
use storage_mode::recorded_storage_mode_tag;

mod fast_load_sidecars;
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

// ─── Save ────────────────────────────────────────────────────────────────────

/// Stamp save metadata and snapshot index keys. Quick, runs with GIL held.
pub fn prepare_save(graph: &mut Arc<DirGraph>) {
    let g = crate::graph::handle::make_dir_graph_mut_preserving_lineage(graph);
    g.save_metadata = SaveMetadata::current();
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

fn zstd_decompress(data: &[u8]) -> io::Result<Vec<u8>> {
    zstd_decompress_limited(data, MAX_DECOMPRESSED_SECTION_BYTES)
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
            serde_codec::CodecVersion::from_tag(codec_tag).map_err(io::Error::other)?,
            payload,
            allocated_bytes,
            serde_codec::DecodeLimits::new(MAX_CODEC_BYTES, MAX_CODEC_BYTES),
        )
        .map_err(io::Error::other);
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

fn zstd_decompress_limited(data: &[u8], limit: u64) -> io::Result<Vec<u8>> {
    let decoder = zstd::Decoder::new(std::io::Cursor::new(data))
        .map_err(|e| invalid_data(format!("invalid zstd section: {e}")))?;
    let mut bounded = decoder.take(limit.saturating_add(1));
    let mut decoded = Vec::new();
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

/// The `<name>.kgl.tmp.` prefix every in-flight save writes under, for `dest`.
///
/// Single source of truth for the shape: [`write_kgl_with`] appends
/// `<pid>.<nonce>` to it, and [`reap_stale_save_temps`] parses the same two
/// fields back out. A drift between the two would turn the reaper into a
/// silent no-op — the failure mode it exists to fix.
fn save_temp_prefix(dest: &Path) -> String {
    format!(
        "{}.tmp.",
        dest.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "graph.kgl".to_string()),
    )
}

/// How long a temp whose owner cannot be identified is kept before it is
/// treated as abandoned.
///
/// Only reached where process liveness is unavailable (non-Unix, or a `kill`
/// that answers neither "alive" nor "gone"). Long enough that no plausible
/// save is still running, short enough that the litter does not accumulate
/// across a machine's lifetime.
const UNIDENTIFIED_TEMP_MAX_AGE: Duration = Duration::from_secs(24 * 60 * 60);

/// Delete `<name>.tmp.<pid>.<nonce>` siblings of `path` whose writing process
/// is gone, and report how many were removed.
///
/// [`write_kgl_with`] removes its own temp on every error path it can see, but
/// it cannot see the one that matters: a `SIGKILL`, an OOM-kill or a power cut
/// part-way through a save leaves a **full-size** copy of the graph beside the
/// real file, and nothing else deletes it — a crash-looping writer over a
/// multi-GB graph fills the volume with copies of itself.
///
/// **Nothing a live process could still be writing is touched.** A temp is
/// removed only when its embedded pid is provably not a running process (or,
/// where that cannot be established, when the file has not been touched for
/// [`UNIDENTIFIED_TEMP_MAX_AGE`]); this process's own pid is always skipped,
/// since a concurrent save on another thread is exactly the in-flight case.
/// Pid reuse can only make the reaper *keep* a file it could have deleted,
/// which leaks a temp rather than destroying a live save.
///
/// Best-effort: nothing depends on the reap having happened, so an unreadable
/// directory or a failed unlink is not reported.
pub fn reap_stale_save_temps(path: &Path) -> usize {
    let prefix = save_temp_prefix(path);
    let dir = match path.parent().filter(|p| !p.as_os_str().is_empty()) {
        Some(d) => d.to_path_buf(),
        None => Path::new(".").to_path_buf(),
    };
    let entries = match std::fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(_) => return 0,
    };
    let mut reaped = 0;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(pid) = temp_owner_pid(name, &prefix) else {
            continue;
        };
        if pid == std::process::id() {
            continue;
        }
        let abandoned = match process_is_alive(pid) {
            Some(alive) => !alive,
            None => temp_is_older_than(&entry, UNIDENTIFIED_TEMP_MAX_AGE),
        };
        if abandoned && std::fs::remove_file(entry.path()).is_ok() {
            reaped += 1;
        }
    }
    reaped
}

/// The pid embedded in `name`, when `name` is a save temp of `prefix`'s graph.
///
/// Both trailing fields must parse: a file that merely *starts* with the
/// prefix (`app.kgl.tmp.notes`) is somebody else's, and deleting it because it
/// shared a prefix would be the reaper causing the data loss it prevents.
fn temp_owner_pid(name: &str, prefix: &str) -> Option<u32> {
    let rest = name.strip_prefix(prefix)?;
    let (pid, nonce) = rest.split_once('.')?;
    nonce.parse::<u64>().ok()?;
    let pid: u32 = pid.parse().ok()?;
    (pid > 0).then_some(pid)
}

/// Whether `entry` was last modified longer than `max_age` ago. An
/// unreadable timestamp answers "no" — the reaper's default is always to keep.
fn temp_is_older_than(entry: &std::fs::DirEntry, max_age: Duration) -> bool {
    entry
        .metadata()
        .and_then(|m| m.modified())
        .ok()
        .and_then(|modified| SystemTime::now().duration_since(modified).ok())
        .is_some_and(|age| age > max_age)
}

/// Whether process `pid` exists, or `None` when this platform cannot say.
///
/// `kill(pid, 0)` sends no signal — it performs only the existence and
/// permission checks — so it is the standard liveness probe. `EPERM` means the
/// process exists and belongs to another user, which is still "alive"; `ESRCH`
/// is the only answer that licenses a delete.
#[cfg(unix)]
fn process_is_alive(pid: u32) -> Option<bool> {
    // SAFETY: `kill` with signal 0 performs no action beyond the existence and
    // permission check, and `pid > 0` (checked by `temp_owner_pid`) keeps it
    // from addressing a process *group*.
    if unsafe { libc::kill(pid as libc::pid_t, 0) } == 0 {
        return Some(true);
    }
    match std::io::Error::last_os_error().raw_os_error() {
        Some(libc::ESRCH) => Some(false),
        Some(libc::EPERM) => Some(true),
        _ => None,
    }
}

/// Non-Unix platforms have no cheap equivalent, so the age fallback owns the
/// decision there.
#[cfg(not(unix))]
fn process_is_alive(_pid: u32) -> Option<bool> {
    None
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
    let write_result = (|| -> io::Result<()> {
        let file = File::create(&tmp)?;
        let mut writer = BufWriter::new(file);
        write_kgl_to(graph, &mut writer)?;
        writer.flush()?;
        let file = writer
            .into_inner()
            .map_err(|e| io::Error::other(e.to_string()))?;
        if fsync {
            file.sync_all()?;
        }
        Ok(())
    })();

    // On any write error, remove the temp so a failed save leaves no litter.
    if let Err(e) = write_result {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }

    if let Err(e) = std::fs::rename(&tmp, dest) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }

    // fsync the directory so the rename itself is durable (the rename can
    // otherwise be lost on a crash even though the file bytes are synced).
    if fsync {
        if let Some(d) = dir {
            if let Ok(dirfile) = File::open(d) {
                let _ = dirfile.sync_all();
            }
        }
    }
    Ok(())
}

/// Serialize, compress, and write the graph to a `.kgl` file, atomically
/// and durably (temp + fsync + rename — see [`write_kgl_with`]). Heavy
/// I/O, safe to run without the GIL.
///
/// The bytes are the v6 container: `V6_MAGIC`, an explicit Postcard codec
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
    optional: [(&str, Option<&[u8]>); 5],
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
    validate_column_keys_registered(graph)?;
    let codec = serde_codec::CodecVersion::PostcardV1;

    // 1. Serialize topology with node properties stripped into column sections.
    let topology_raw = {
        let _strip = StripPropertiesGuard::new();
        let _guard = SerdeSerializeGuard::new(&graph.interner);
        codec_ser(codec, &graph.graph)?
    };
    let topology_compressed = zstd_compress(&topology_raw)?;
    drop(topology_raw); // free before compressing columns

    // 2. Column sections, one per node type, sorted by type_name: the backend's
    // map is a HashMap whose per-instance RandomState would otherwise vary the
    // section order across processes, breaking the byte-level reproducibility
    // the `test_phase4_parity` golden-hash test relies on. Sorting is free
    // (type_name count is small) and doesn't affect the format: each section
    // is self-describing and the decoder iterates column_sections_meta in order.
    let mut column_sections_meta: Vec<PortableColumnSection> = Vec::new();
    let mut column_sections_data: Vec<Vec<u8>> = Vec::new();

    let mut column_stores_sorted: Vec<(&str, &Arc<ColumnStore>)> = graph.column_stores_by_name();
    column_stores_sorted.sort_by(|a, b| a.0.cmp(b.0));
    for (type_name, store) in column_stores_sorted {
        let packed = store.write_packed_with_codec(
            &graph.interner,
            codec,
            // v6: integer columns pick their smaller encoding per column.
            crate::graph::storage::packed_codec::IntColumnEncoding::Auto,
        )?;
        let compressed = zstd_compress(&packed)?;
        drop(packed); // free uncompressed before next type

        let mut cols = HashMap::new();
        for (slot, ik) in store.schema().iter() {
            let prop_name = graph.interner.resolve(ik);
            if let Some(col) = store.column(slot as usize) {
                // The *logical* column type. The per-column encoding actually
                // used lives in the section itself (a v6 `Int64` column may be
                // written delta-varint); the loader reads the section's tag and
                // uses these entries only for their key set.
                cols.insert(prop_name.to_string(), col.type_tag().to_string());
            }
        }

        column_sections_meta.push(PortableColumnSection {
            type_name: type_name.to_string(),
            compressed_size: compressed.len() as u64,
            row_count: store.row_count(),
            columns: cols,
        });
        column_sections_data.push(compressed);
    }

    // 3. Embeddings, through a BTreeMap view: `graph.embeddings` is a HashMap
    // whose per-process RandomState would otherwise randomize entry order,
    // breaking the byte-reproducibility the column sections above already
    // guarantee (same wire shape, HashMap deserializes it unchanged).
    let embedding_compressed = if !graph.embeddings.is_empty() {
        let ordered: std::collections::BTreeMap<_, _> = graph.embeddings.iter().collect();
        let raw = codec_ser(codec, &ordered)?;
        Some(zstd_compress(&raw)?)
    } else {
        None
    };

    // 4. Compress timeseries if any (BTreeMap view for the same reason).
    let timeseries_compressed = if !graph.timeseries_store.is_empty() {
        let ordered: std::collections::BTreeMap<_, _> = graph.timeseries_store.iter().collect();
        let raw = codec_ser(codec, &ordered)?;
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

    let vector_index_compressed = match encode_vector_indexes(graph)? {
        Some(payload) => Some(zstd_compress(&payload)?),
        None => None,
    };

    let text_index_compressed = match encode_text_indexes(graph)? {
        Some(payload) => Some(zstd_compress(&payload)?),
        None => None,
    };

    let section_digests = build_section_digests(
        &topology_compressed,
        &column_sections_meta,
        &column_sections_data,
        [
            (EMBEDDINGS_SECTION, embedding_compressed.as_deref()),
            (TIMESERIES_SECTION, timeseries_compressed.as_deref()),
            (
                SECONDARY_LABELS_SECTION,
                secondary_labels_compressed.as_deref(),
            ),
            (VECTOR_INDEX_SECTION, vector_index_compressed.as_deref()),
            (TEXT_INDEX_SECTION, text_index_compressed.as_deref()),
        ],
    );
    let mut metadata = FileMetadata::from_graph(graph);
    metadata.section_digests = section_digests;
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
    metadata.vector_index_compressed_size = vector_index_compressed
        .as_ref()
        .map(|b| b.len() as u64)
        .unwrap_or(0);
    metadata.text_index_compressed_size = text_index_compressed
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

    // Header: magic (4B) + codec (1B) + core_data_version (4B) +
    // metadata_length (4B). The codec byte prevents implicit byte sniffing.
    writer.write_all(&V6_MAGIC)?;
    writer.write_all(&[codec.tag()])?;
    writer.write_all(&CURRENT_CORE_DATA_VERSION.to_le_bytes())?;
    writer.write_all(&(metadata_json.len() as u32).to_le_bytes())?;
    writer.write_all(&metadata_json)?;

    writer.write_all(&topology_compressed)?;

    // Column sections in metadata order.
    for section_data in &column_sections_data {
        writer.write_all(section_data)?;
    }

    if let Some(emb_data) = &embedding_compressed {
        writer.write_all(emb_data)?;
    }

    if let Some(ts_data) = &timeseries_compressed {
        writer.write_all(ts_data)?;
    }

    // Secondary-label-index section (0.10.5+). Single-label graphs
    // skip this entirely (encode returned None).
    if let Some(sl_data) = &secondary_labels_compressed {
        writer.write_all(sl_data)?;
    }

    // HNSW vector-index section (0.11.0+). Omitted when no store is indexed.
    if let Some(vi_data) = &vector_index_compressed {
        writer.write_all(vi_data)?;
    }

    // BM25 text-index section (0.16.10+). Omitted when no type is indexed —
    // which is also why its metadata key is skipped at zero: a graph with no
    // text index writes pre-0.16.10 bytes.
    if let Some(ti_data) = &text_index_compressed {
        writer.write_all(ti_data)?;
    }

    // Flush the writer's own buffer. The atomic-save wrapper additionally
    // fsyncs the underlying file; for an in-memory `Vec<u8>` writer this is
    // a harmless no-op.
    writer.flush()?;
    Ok(())
}

/// Everything a `.kgl` write needs done to the graph before its bytes are
/// produced: stamp the save metadata ([`prepare_save`]), then run the
/// consolidation pass that reclaims rows deleted nodes left behind, restores
/// ascending row order, and re-derives each column's type from its type's
/// metadata. Row order *is* the file's node binding, so a write that skips
/// this can serialize every row against the wrong node.
///
/// This is the single pre-write step for every `.kgl` producer — the
/// path-writing [`save_inmemory_with`] and the buffer-writing bindings
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
pub fn save_inmemory_with(graph: &mut Arc<DirGraph>, path: &str, fsync: bool) -> io::Result<()> {
    prepare_kgl_write(graph);
    write_kgl_with(graph, path, fsync)
}

/// Mode-aware durable save: dispatches to `DirGraph::save_disk` for
/// disk-backed graphs, `save_inmemory_with` otherwise. This is THE single
/// save-dispatch — the wheel (`KnowledgeGraph::save`), the MCP server,
/// and the C ABI (`kglite_save_graph`) all route through it so dispatch
/// + durability behaviour can't drift between bindings.
pub fn save_graph(graph: &mut Arc<DirGraph>, path: &str) -> Result<(), SaveError> {
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
    save_guard::ensure_target_recovered(graph, path)?;
    if graph.graph.is_disk() {
        let dir = crate::graph::handle::make_dir_graph_mut_preserving_lineage(graph);
        return dir.save_disk(path).map_err(SaveError::Io);
    }
    save_inmemory_with(graph, path, fsync).map_err(|e| SaveError::Io(e.to_string()))
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

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn validate_and_rebuild_embedding_norms(
    embeddings: &mut HashMap<(String, String), EmbeddingStore>,
) -> io::Result<()> {
    for store in embeddings.values_mut() {
        store.validate_shape().map_err(invalid_data)?;
        store.rebuild_norms();
    }
    Ok(())
}

pub(crate) fn pre_014_bincode_error(artifact: &str) -> io::Error {
    invalid_data(format!(
        "Unsupported pre-0.14 bincode persistence: {artifact}. This build reads Postcard \
         persistence only. Open the artifact with kglite 0.13.4 and re-save or re-export it, \
         then retry; alternatively rebuild it from the original source."
    ))
}

struct SectionCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
    /// The writer's per-section CRC32 digests, keyed by canonical section name
    /// (`FileMetadata::section_digests`). Empty for a file written before that
    /// field existed — [`Self::take`] then verifies nothing, which is exactly
    /// how those files loaded before.
    digests: BTreeMap<String, u32>,
}

impl<'a> SectionCursor<'a> {
    fn new(bytes: &'a [u8], offset: usize, digests: BTreeMap<String, u32>) -> io::Result<Self> {
        if offset > bytes.len() {
            return Err(invalid_data("section cursor starts past end of file"));
        }
        Ok(Self {
            bytes,
            offset,
            digests,
        })
    }

    /// Slice the next `encoded_len` bytes and check them against the digest
    /// recorded for `section`, if the file carries one.
    ///
    /// This is *the* integrity gate for the container payload: every section
    /// is read through here, and the check happens before the bytes reach a
    /// decoder. `section` is both the digest key and the name errors report,
    /// so a mismatch tells the operator which part of the file is damaged.
    fn take(&mut self, encoded_len: u64, section: &str) -> io::Result<&'a [u8]> {
        let len = usize::try_from(encoded_len)
            .map_err(|_| invalid_data(format!("{section} section size does not fit usize")))?;
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| invalid_data(format!("{section} section offset overflow")))?;
        let bytes = self.bytes.get(self.offset..end).ok_or_else(|| {
            invalid_data(format!(
                "file is truncated — {section} section needs {len} bytes at offset {}",
                self.offset
            ))
        })?;
        self.verify(section, bytes)?;
        self.offset = end;
        Ok(bytes)
    }

    fn verify(&self, section: &str, bytes: &[u8]) -> io::Result<()> {
        let Some(&expected) = self.digests.get(section) else {
            return Ok(());
        };
        let actual = section_digest(bytes);
        if actual != expected {
            return Err(invalid_data(format!(
                "the '{section}' section of this .kgl file is corrupt — it does not match the \
                 CRC32 digest recorded when the file was written (recorded {expected:#010x}, \
                 computed {actual:#010x}). Restore the file from a backup or rebuild the graph \
                 from its source."
            )));
        }
        Ok(())
    }
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
pub fn load_file(path: &str) -> io::Result<Arc<DirGraph>> {
    let p = std::path::Path::new(path);
    if p.is_dir() {
        return load_disk_dir(p);
    }

    let file = File::open(path)?;
    let file_len = file.metadata()?.len();

    // For large files, mmap avoids the full copy into a Vec<u8>
    if file_len >= FILE_MMAP_THRESHOLD {
        // SAFETY: standalone `.kgl` files follow a caller-enforced
        // single-writer contract. Writers replace the destination atomically
        // rather than truncating it in place, so this opened inode remains
        // stable for the mapping's lifetime.
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < 4 {
            return Err(io::Error::other(
                "File is too small to be a valid kglite file.",
            ));
        }
        if mmap[..4] == V6_MAGIC {
            return load_portable_container(&mmap, "v6");
        }
        if mmap[..4] == V5_MAGIC {
            return load_portable_container(&mmap, "v5");
        }
        if mmap[..4] == V4_MAGIC {
            return Err(pre_014_bincode_error(".kgl container v4"));
        }
        if mmap[..4] == V3_MAGIC {
            return Err(io::Error::other(V3_HARD_BREAK_MSG));
        }
        if mmap[..3] == V6_MAGIC[..3] && mmap[3] > V6_MAGIC[3] {
            return Err(newer_portable_format_error(mmap[3]));
        }
        return Err(unrecognized_magic_error(&mmap[..4], &format!("'{path}'")));
    }

    let buf = std::fs::read(path)?;
    if buf.len() < 4 {
        return Err(io::Error::other(
            "File is too small to be a valid kglite file.",
        ));
    }
    if buf[..4] == V6_MAGIC {
        load_portable_container(&buf, "v6")
    } else if buf[..4] == V5_MAGIC {
        load_portable_container(&buf, "v5")
    } else if buf[..4] == V4_MAGIC {
        Err(pre_014_bincode_error(".kgl container v4"))
    } else if buf[..4] == V3_MAGIC {
        Err(io::Error::other(V3_HARD_BREAK_MSG))
    } else if buf[..3] == V6_MAGIC[..3] && buf[3] > V6_MAGIC[3] {
        Err(newer_portable_format_error(buf[3]))
    } else {
        Err(unrecognized_magic_error(&buf[..4], &format!("'{path}'")))
    }
}

/// Load an in-memory graph from a `.kgl` byte buffer — the counterpart of
/// [`write_kgl_to`] / `KnowledgeGraph.to_bytes()`. Same magic/version
/// validation and error classification as [`load_file`]'s small-file branch.
/// Disk-mode graphs are a directory, not a byte stream, so this only handles
/// the single-file in-memory format.
///
/// **It does not read the `.kgl` from disk — the caller already holds the
/// bytes. It is not filesystem-free.** Loading a graph that has column
/// sections creates a per-process spill directory
/// (`$TMPDIR/kglite_portable_<pid>_<nanos>/type_<n>/`) and writes any column
/// blob of 256 KB or more into it so the column can be mmap'd instead of
/// heap-allocated; smaller columns stay on the heap and the directory is left
/// empty. The paths are registered on the returned graph, and the last
/// `DirGraph` holding them removes the tree in `Drop` — so they live exactly
/// as long as the graph, and a process killed before that drop leaves them
/// behind for the OS temp sweep.
pub fn load_kgl_bytes(data: &[u8]) -> io::Result<Arc<DirGraph>> {
    if data.len() < 4 {
        return Err(io::Error::other(
            "Byte buffer is too small to be a valid kglite graph.",
        ));
    }
    if data[..4] == V6_MAGIC {
        load_portable_container(data, "v6")
    } else if data[..4] == V5_MAGIC {
        load_portable_container(data, "v5")
    } else if data[..4] == V4_MAGIC {
        Err(pre_014_bincode_error(".kgl container v4"))
    } else if data[..4] == V3_MAGIC {
        Err(io::Error::other(V3_HARD_BREAK_MSG))
    } else if data[..3] == V6_MAGIC[..3] && data[3] > V6_MAGIC[3] {
        Err(newer_portable_format_error(data[3]))
    } else {
        Err(unrecognized_magic_error(&data[..4], "the byte buffer"))
    }
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

    // Backfill the connection_types O(1)-lookup cache from the loaded metadata.
    // Left empty, `has_connection_type`'s metadata-fallback branch is correct on
    // a freshly-loaded graph but flips into the wrong branch the moment anything
    // calls `register_connection_type` (which inserts into the cache and trips
    // the "use cache" fast path on subsequent lookups).
    graph.build_connection_types_cache();

    log_stage("load_disk_dir_total", _load_t);

    Ok(Arc::new(graph))
}

/// Install a disk graph's column stores — the mmap-backed `columns.bin` +
/// `columns_meta` pair when present, otherwise the per-type
/// `columns/<type>/columns.zst` sidecars. Split out of `load_disk_dir` to keep
/// it under the function-complexity ceiling; cold load-time path.
fn load_disk_column_stores(dir: &std::path::Path, graph: &mut DirGraph) -> io::Result<()> {
    use crate::graph::io::load_timing::{log_stage, stage_timer};

    // Prefer the mmap-backed pair (columns.bin + columns_meta), checking
    // `seg_000/` as well as the directory root: a later layout moved these
    // files into `seg_000/`, and without both locations the load fell through
    // to the per-type `columns/<type>/columns.zst` branch, which returned an
    // empty `column_stores` map and broke `MATCH (n:Type)` queries after a
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
        // SAFETY: GraphDirectoryLock serializes disk-graph writers, which
        // publish a new immutable generation instead of truncating the
        // generation selected by this reader. This columns.bin inode remains
        // stable for the mapping's lifetime.
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        let mmap_arc = std::sync::Arc::new(mmap);

        // Prefer the binary sidecar over JSON (slow for 295 MB).
        let type_metas: Vec<ColumnTypeMeta> = if meta_bin_path.exists() {
            let compressed = std::fs::read(&meta_bin_path)?;
            let bytes = zstd_decompress(&compressed)?;
            decode_disk_serde(&bytes, bytes.capacity() as u64)?
        } else {
            let meta_json = std::fs::read_to_string(&meta_json_path)?;
            serde_json::from_str(&meta_json).map_err(io::Error::other)?
        };

        // `columns.bin` bytes are untrusted disk input, but the hot string
        // readers use `from_utf8_unchecked` (see MmapColumnStore::read_str).
        // Validate every string column once here — load-time, amortized —
        // so the per-access unchecked conversion stays sound. Opt-out for
        // very large trusted graphs (validation touches every string byte,
        // forcing a full read of columns.bin): KGLITE_SKIP_UTF8_VALIDATION=1.
        let skip_utf8 = std::env::var_os("KGLITE_SKIP_UTF8_VALIDATION").is_some();
        for tm in type_metas {
            let store = tm.to_mmap_store(std::sync::Arc::clone(&mmap_arc));
            if !skip_utf8 {
                store.validate_utf8(&tm.type_name)?;
            }
            let cs = crate::graph::storage::column_store::ColumnStore::from_mmap_store(
                std::sync::Arc::new(store),
            );
            graph.install_column_store(&tm.type_name, Arc::new(cs));
        }

        // Additively load sidecars for types added post-`load_ntriples`
        // via `add_nodes`. The sidecar writer in `DirGraph::save_disk`
        // emits `columns/<type>/columns.zst` only for types NOT in
        // `columns_meta`, so the two paths don't clash — but we still
        // check before overwriting out of caution.
        load_column_sidecars(dir, graph)?;
    } else {
        load_column_sidecars(dir, graph)?;
    }
    log_stage("column_stores_load", t);
    Ok(())
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

/// Decode a v5 or v6 container. The two share a header, a codec and a section
/// layout; they differ only in which per-column encodings the column sections
/// may use, and the column reader dispatches on the section's own type tags.
/// `format_name` is the version the caller matched, and appears in errors.
fn load_portable_container(buf: &[u8], format_name: &str) -> io::Result<Arc<DirGraph>> {
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
    load_portable_columnar(buf, format_name, codec, core_version, metadata_len, 13)
}

struct PortableSectionPlan {
    columns: Vec<PortableColumnSection>,
    embeddings: u64,
    timeseries: u64,
    secondary_labels: u64,
    vector_index: u64,
    text_index: u64,
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
    let metadata: FileMetadata = serde_json::from_slice(metadata_bytes)
        .map_err(|e| invalid_data(format!("failed to parse {format_name} metadata: {e}")))?;
    if metadata.column_sections.len() > 1_000_000 {
        return Err(invalid_data(format!(
            "{format_name} metadata declares too many column sections"
        )));
    }
    let digests = metadata.section_digests.clone();
    Ok((metadata, SectionCursor::new(buf, metadata_end, digests)?))
}

fn decode_portable_topology(
    codec: serde_codec::CodecVersion,
    sections: &mut SectionCursor<'_>,
    metadata: FileMetadata,
) -> io::Result<(DirGraph, PortableSectionPlan)> {
    let topology_compressed = sections.take(metadata.topology_compressed_size, TOPOLOGY_SECTION)?;
    let topology_raw = zstd_decompress(topology_compressed)?;
    let mut interner = StringInterner::new();
    let graph: crate::graph::schema::GraphBackend = {
        let _guard = SerdeDeserializeGuard::new(&mut interner);
        codec_deser(codec, &topology_raw, topology_raw.capacity() as u64)?
    };
    let plan = PortableSectionPlan {
        columns: metadata.column_sections.clone(),
        embeddings: metadata.embeddings_compressed_size,
        timeseries: metadata.timeseries_compressed_size,
        secondary_labels: metadata.secondary_labels_compressed_size,
        vector_index: metadata.vector_index_compressed_size,
        text_index: metadata.text_index_compressed_size,
    };
    let mut dir_graph = DirGraph::from_graph(graph);
    dir_graph.interner = interner;
    metadata.apply_to(&mut dir_graph);
    dir_graph.rebuild_type_indices_and_schemas();
    dir_graph.build_connection_types_cache();
    Ok((dir_graph, plan))
}

fn portable_temp_dir() -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "kglite_portable_{}_{:x}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ))
}

fn load_portable_column_section(
    codec: serde_codec::CodecVersion,
    dir_graph: &mut DirGraph,
    sections: &mut SectionCursor<'_>,
    section_meta: &PortableColumnSection,
    section_index: usize,
    temp_dir: &Path,
) -> io::Result<()> {
    let compressed = sections.take(
        section_meta.compressed_size,
        &column_section_key(&section_meta.type_name),
    )?;
    let packed = zstd_decompress(compressed)?;
    let expected_rows = dir_graph
        .type_indices
        .get(&section_meta.type_name)
        .map_or(0, |nodes| nodes.len());
    if section_meta.row_count as usize != expected_rows {
        return Err(invalid_data(format!(
            "column section {section_index} for '{}' declares {} rows; topology has {expected_rows}",
            section_meta.type_name, section_meta.row_count
        )));
    }
    // Column slot order comes from the PAYLOAD, not from `section_meta.columns`.
    //
    // The packed block is self-describing and ordered — it carries
    // `(name, type_tag, data)` per column in the writing store's slot order —
    // whereas `section_meta.columns` is a `HashMap` that records only the key
    // set and type tags (see the writer's own note that the loader "uses these
    // entries only for their key set"). Building the schema from that map made
    // slot order a `RandomState` artefact that differed on every load, so
    // re-saving a file produced different bytes each run.
    //
    // Reading the recorded order makes a re-save byte-identical to the
    // original, not merely deterministic.
    let mut ordered_names = ColumnStore::packed_column_names(&packed)?;
    // A key the metadata declares but the payload does not carry has no
    // recorded position; append such keys by name so they still get a slot and
    // the result stays deterministic. Expected to be empty — writer and payload
    // are built from the same schema — but silently dropping a declared column
    // would be worse than an arbitrary-but-stable position.
    let in_payload: std::collections::HashSet<&str> =
        ordered_names.iter().map(String::as_str).collect();
    let mut orphans: Vec<&String> = section_meta
        .columns
        .keys()
        .filter(|name| !in_payload.contains(name.as_str()))
        .collect();
    orphans.sort();
    let orphans: Vec<String> = orphans.into_iter().cloned().collect();
    ordered_names.extend(orphans);

    let col_keys = ordered_names
        .iter()
        .map(|name| {
            dir_graph
                .interner
                .try_get_or_intern(name)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
        })
        .collect::<io::Result<Vec<_>>>()?;
    let column_schema = Arc::new(crate::graph::schema::TypeSchema::from_keys(col_keys));
    let type_meta = dir_graph
        .node_type_metadata
        .get(&section_meta.type_name)
        .cloned()
        .unwrap_or_default();
    let type_temp_dir = temp_dir.join(format!("type_{section_index}"));
    std::fs::create_dir_all(&type_temp_dir)?;
    let store = ColumnStore::load_packed_with_codec(
        column_schema,
        &type_meta,
        &dir_graph.interner,
        &packed,
        section_meta.row_count,
        Some(&type_temp_dir),
        codec,
    )?;
    dir_graph.install_column_store(&section_meta.type_name, Arc::new(store));
    Ok(())
}

fn load_portable_columns(
    codec: serde_codec::CodecVersion,
    dir_graph: &mut DirGraph,
    sections: &mut SectionCursor<'_>,
    columns: &[PortableColumnSection],
) -> io::Result<()> {
    let temp_dir = portable_temp_dir();
    if let Ok(mut dirs) = dir_graph.temp_dirs.lock() {
        dirs.push(temp_dir.clone());
    }
    for (index, metadata) in columns.iter().enumerate() {
        load_portable_column_section(codec, dir_graph, sections, metadata, index, &temp_dir)?;
    }
    attach_portable_column_stores(dir_graph);
    Ok(())
}

fn load_portable_optional_sections(
    codec: serde_codec::CodecVersion,
    core_version: u32,
    dir_graph: &mut DirGraph,
    sections: &mut SectionCursor<'_>,
    plan: &PortableSectionPlan,
) -> io::Result<()> {
    if plan.embeddings > 0 {
        if core_version < EMBED_PROVENANCE_MIN_VERSION {
            return Err(io::Error::other(EMBED_FORMAT_BREAK_MSG));
        }
        let compressed = sections.take(plan.embeddings, EMBEDDINGS_SECTION)?;
        let raw = zstd_decompress(compressed)?;
        let mut embeddings: HashMap<(String, String), EmbeddingStore> =
            codec_deser(codec, &raw, raw.capacity() as u64)?;
        validate_and_rebuild_embedding_norms(&mut embeddings)?;
        dir_graph.embeddings = embeddings;
    }
    if plan.timeseries > 0 {
        let compressed = sections.take(plan.timeseries, TIMESERIES_SECTION)?;
        let raw = zstd_decompress(compressed)?;
        dir_graph.timeseries_store = codec_deser(codec, &raw, raw.capacity() as u64)?;
    }
    if plan.secondary_labels > 0 {
        let compressed = sections.take(plan.secondary_labels, SECONDARY_LABELS_SECTION)?;
        let raw = zstd_decompress(compressed)?;
        decode_secondary_label_index(&raw, dir_graph)?;
    }
    if plan.vector_index > 0 {
        // Framing failures propagate — a section that is truncated or fails
        // its digest means the *file* is damaged, and skipping it would leave
        // the reader with no signal that anything was wrong. The payload
        // itself stays optional: it is self-describing and rebuildable, so an
        // index this build does not recognise is still skipped silently (see
        // `decode_vector_indexes`).
        let compressed = sections.take(plan.vector_index, VECTOR_INDEX_SECTION)?;
        if let Ok(raw) = zstd_decompress(compressed) {
            decode_vector_indexes(&raw, dir_graph);
        }
    }
    if plan.text_index > 0 {
        // Same split as the vector section above: framing failures are file
        // damage and propagate; the payload itself is a rebuildable cache and
        // an unreadable one is skipped silently (see `decode_text_indexes`).
        let compressed = sections.take(plan.text_index, TEXT_INDEX_SECTION)?;
        if let Ok(raw) = zstd_decompress(compressed) {
            decode_text_indexes(&raw, dir_graph);
        }
    }
    Ok(())
}

/// Load the shared v5/v6 columnar section layout through the codec selected by
/// the already-validated container header.
fn load_portable_columnar(
    buf: &[u8],
    format_name: &str,
    codec: serde_codec::CodecVersion,
    core_version: u32,
    metadata_len: usize,
    metadata_start: usize,
) -> io::Result<Arc<DirGraph>> {
    if metadata_len > MAX_METADATA_BYTES {
        return Err(invalid_data(format!(
            "{format_name} metadata is {metadata_len} bytes; limit is {MAX_METADATA_BYTES}"
        )));
    }

    if core_version > CURRENT_CORE_DATA_VERSION {
        return Err(io::Error::other(format!(
            "File uses core data version {} but this library only supports up to version {}. \
             Please upgrade kglite.",
            core_version, CURRENT_CORE_DATA_VERSION,
        )));
    }
    let (metadata, mut sections) =
        parse_portable_metadata(buf, format_name, metadata_len, metadata_start)?;
    // Resolved before a section is decompressed, so an unplaceable mode fails
    // before the expensive part.
    let recorded_mode = metadata.portable_storage_mode()?;
    let (mut dir_graph, plan) = decode_portable_topology(codec, &mut sections, metadata)?;
    load_portable_columns(codec, &mut dir_graph, &mut sections, &plan.columns)?;
    load_portable_optional_sections(codec, core_version, &mut dir_graph, &mut sections, &plan)?;
    // Honour the recorded mode. The payload always deserializes into a memory
    // backend (`GraphBackend`'s Deserialize), so a mapped-saved checkpoint is
    // swapped onto the mapped backend here — after the graph is complete, and
    // by moving the topology rather than copying it. Memory (and a file that
    // recorded nothing) is already what the decode produced, so it is a no-op.
    crate::graph::storage::mode::convert_dir_graph_to_mode(&mut dir_graph, recorded_mode)
        .map_err(io::Error::other)?;
    Ok(Arc::new(dir_graph))
}

mod columns;
use columns::{attach_portable_column_stores, load_column_sidecars};

mod save_guard;
pub use save_guard::SaveError;

mod text_index_persistence;
use text_index_persistence::{decode_text_indexes, encode_text_indexes};

mod vector_persistence;

#[allow(unused_imports)]
pub use vector_persistence::ExportStats;
use vector_persistence::{decode_vector_indexes, encode_vector_indexes};
pub use vector_persistence::{
    export_embeddings_to_file, import_embeddings_from_file, EmbeddingExportFilter, ImportStats,
};
#[cfg(test)]
#[path = "file_tests.rs"]
mod file_tests;
