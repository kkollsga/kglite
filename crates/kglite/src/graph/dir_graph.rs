//! DirGraph — transactional container for the in-memory graph.
//!
//! Owns the `StableDiGraph` + all type/property/composite/range indexes,
//! OCC `version`, `schema_locked`, spatial / temporal / timeseries configs,
//! embedding stores, connection-type metadata, and schema definitions.

use crate::datatypes::values::Value;
use crate::graph::schema::{
    CompositeIndexKey, CompositeValue, ConnectionTypeInfo, ConnectivityTriple, EdgeData,
    EmbeddingStore, GraphBackend, IndexKey, InternedKey, NodeData, PropertyStorage, SaveMetadata,
    SchemaDefinition, SpatialConfig, StringInterner, TemporalConfig, TypeIdIndex, TypeSchema,
};
use crate::graph::storage::disk::id_index::IdIndexStore;
use crate::graph::storage::disk::type_index::TypeIndexStore;
use crate::graph::storage::{GraphRead, GraphWrite, MemoryGraph};
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::stable_graph::StableDiGraph;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

/// Core graph storage: a directed graph (petgraph `StableDiGraph`) with fast
/// type-based indexing and optional property/composite/range/spatial indexes.
///
/// Fields include `type_indices` for O(1) node-type lookup, `property_indices`
/// for indexed equality filters, connection-type metadata, schema definitions,
/// and optional embedding stores for vector similarity search.
#[derive(Clone, Serialize, Deserialize)]
pub struct DirGraph {
    pub graph: GraphBackend,
    /// Skipped during serialization — rebuilt from graph on load via `rebuild_type_indices()`.
    /// On disk graphs the base layer is mmap-backed via `type_indices.bin`;
    /// mutations land in an in-memory overlay.
    #[serde(skip)]
    pub type_indices: TypeIndexStore,
    /// Optional schema definition for validation
    #[serde(default)]
    pub schema_definition: Option<SchemaDefinition>,
    /// Single-property indexes for fast lookups: (node_type, property) -> value -> [node_indices]
    /// Skipped during serialization — rebuilt from `property_index_keys` on load.
    #[serde(skip)]
    pub property_indices: HashMap<IndexKey, HashMap<Value, Vec<NodeIndex>>>,
    /// Composite indexes for multi-field queries: (node_type, [properties]) -> composite_value -> [node_indices]
    /// Skipped during serialization — rebuilt from `composite_index_keys` on load.
    #[serde(skip)]
    pub composite_indices: HashMap<CompositeIndexKey, HashMap<CompositeValue, Vec<NodeIndex>>>,
    /// Persisted list of property index keys so indexes can be rebuilt on load
    #[serde(default)]
    pub property_index_keys: Vec<IndexKey>,
    /// Persisted list of composite index keys so indexes can be rebuilt on load
    #[serde(default)]
    pub composite_index_keys: Vec<CompositeIndexKey>,
    /// B-Tree range indexes for ordered lookups: (node_type, property) -> BTreeMap<Value, [NodeIndex]>
    /// Skipped during serialization — rebuilt from `range_index_keys` on load.
    #[serde(skip)]
    pub range_indices: HashMap<IndexKey, std::collections::BTreeMap<Value, Vec<NodeIndex>>>,
    /// Persisted list of range index keys so indexes can be rebuilt on load
    #[serde(default)]
    pub range_index_keys: Vec<IndexKey>,
    /// Fast O(1) lookup by node ID: node_type -> TypeIdIndex
    /// Lazily built on first use for each node type, skipped during serialization.
    /// Uses compact u32 HashMap when all IDs are UniqueId (e.g., Wikidata mapped mode).
    /// On disk graphs the base layer is mmap-backed via `id_indices.bin`; mutations
    /// land in an in-memory overlay (see `storage/disk/id_index.rs`).
    #[serde(skip)]
    pub id_indices: IdIndexStore,
    /// Fast O(1) lookup for connection types (interned). Populated on first edge access.
    #[serde(skip)]
    pub connection_types: std::collections::HashSet<InternedKey>,
    /// Node type metadata: node_type → { property_name → type_string }
    /// Replaces SchemaNode graph nodes — persisted via serde/bincode.
    #[serde(default)]
    pub node_type_metadata: HashMap<String, HashMap<String, String>>,
    /// Connection type metadata: connection_type → ConnectionTypeInfo
    /// Replaces SchemaNode graph nodes for connections — persisted via serde/bincode.
    #[serde(default)]
    pub connection_type_metadata: HashMap<String, ConnectionTypeInfo>,
    /// Version and library info stamped at save time.
    /// Old files without this field deserialize to SaveMetadata::default() (format_version=0).
    #[serde(default)]
    pub save_metadata: SaveMetadata,
    /// Original ID field name per node type (e.g. "Person" → "npdid").
    /// Stored when the user-supplied unique_id_field differs from "id".
    /// Used for alias resolution: querying by original column name maps to the `id` field.
    #[serde(default)]
    pub id_field_aliases: FxHashMap<String, String>,
    /// Original title field name per node type (e.g. "Person" → "prospect_name").
    /// Stored when the user-supplied node_title_field differs from "title".
    /// Used for alias resolution: querying by original column name maps to the `title` field.
    #[serde(default)]
    pub title_field_aliases: FxHashMap<String, String>,
    /// Parent type for supporting node types: child_type → parent_type.
    /// If a type has an entry here, it is a "supporting" type that belongs to the parent.
    /// Types without an entry are "core" types (shown in describe() inventory).
    #[serde(default)]
    pub parent_types: HashMap<String, String>,
    /// Auto-vacuum threshold: if Some(t), vacuum() is triggered automatically after
    /// DELETE operations when fragmentation_ratio exceeds t and tombstones > 100.
    /// Default: Some(0.3). Set to None to disable.
    #[serde(default = "default_auto_vacuum_threshold")]
    pub auto_vacuum_threshold: Option<f64>,
    /// Spatial configuration per node type: type_name → SpatialConfig.
    /// Declares which properties hold lat/lon or WKT data for auto-resolution.
    #[serde(default)]
    pub spatial_configs: HashMap<String, SpatialConfig>,
    /// Graph-level WKT geometry cache — persists across queries.
    /// Uses Arc<Geometry> to avoid cloning heavy geometry objects.
    /// RwLock allows concurrent reads from parallel row evaluation.
    #[serde(skip)]
    pub wkt_cache: Arc<RwLock<HashMap<String, Arc<geo::Geometry<f64>>>>>,
    /// Lazy edge-type count cache — avoids O(E) rescan for FusedCountEdgesByType.
    /// Invalidated on edge mutations (add/remove).
    #[serde(skip)]
    pub edge_type_counts_cache: Arc<RwLock<Option<HashMap<String, usize>>>>,
    /// Cached type connectivity: (source_type, connection_type, target_type) → count.
    /// Computed by `rebuild_caches()`, persisted in metadata, restored on load.
    /// Invalidated on edge mutations alongside edge_type_counts_cache.
    #[serde(skip)]
    pub type_connectivity_cache: Arc<RwLock<Option<Vec<ConnectivityTriple>>>>,
    /// Columnar embedding storage: (node_type, property_name) -> EmbeddingStore.
    /// Stored separately from NodeData.properties — invisible to normal node API.
    /// Persisted as a separate section in v2 .kgl files.
    #[serde(skip)]
    pub embeddings: HashMap<(String, String), EmbeddingStore>,
    /// Timeseries configuration per node type: type_name → TimeseriesConfig.
    /// Declares composite key labels and known channels for auto-resolution.
    #[serde(default)]
    pub timeseries_configs: HashMap<String, crate::graph::features::timeseries::TimeseriesConfig>,
    /// Per-node timeseries storage: NodeIndex.index() → NodeTimeseries.
    /// Stored separately from NodeData.properties (like embeddings).
    /// Persisted as a separate section in v2 .kgl files.
    #[serde(skip)]
    pub timeseries_store: HashMap<usize, crate::graph::features::timeseries::NodeTimeseries>,
    /// Temporal configuration per node type: type_name → TemporalConfig.
    /// Nodes of this type are auto-filtered by validity period in select().
    #[serde(default)]
    pub temporal_node_configs: HashMap<String, TemporalConfig>,
    /// Temporal configurations per connection type: connection_type → Vec<TemporalConfig>.
    /// Multiple configs per type support shared connection type names across source types
    /// (e.g., HAS_LICENSEE used by Field, Licence, BusinessArrangement with different field names).
    /// Edges of this type are auto-filtered by validity period in traverse().
    #[serde(default)]
    pub temporal_edge_configs: HashMap<String, Vec<TemporalConfig>>,
    /// Per-type columnar property stores. When populated, nodes of these types
    /// use `PropertyStorage::Columnar` instead of `Compact`.
    /// Not persisted — rebuilt on load if columnar mode is enabled.
    #[serde(skip)]
    pub column_stores: HashMap<String, Arc<crate::graph::storage::column_store::ColumnStore>>,
    /// Memory limit for columnar heap storage. If Some(n), `enable_columnar()`
    /// will spill columns to temp files when total heap_bytes exceeds n.
    #[serde(skip)]
    pub memory_limit: Option<usize>,
    /// Directory for spill files. Defaults to std::env::temp_dir()/kglite_spill_<pid>.
    #[serde(skip)]
    pub spill_dir: Option<std::path::PathBuf>,
    /// Temp directories created during load or spill that should be cleaned up on drop.
    /// Uses Arc so clones share ownership — only the last clone cleans up.
    #[serde(skip)]
    pub(crate) temp_dirs: Arc<std::sync::Mutex<Vec<std::path::PathBuf>>>,
    /// If true, Cypher mutations (CREATE, SET, DELETE, REMOVE, MERGE) are rejected
    /// and describe() omits mutation documentation.
    #[serde(skip)]
    pub read_only: bool,
    /// If true, Cypher mutations (CREATE, SET, MERGE) are validated against
    /// the frozen schema (node_type_metadata + connection_type_metadata).
    /// Unlike read_only, mutations are still allowed — they just must conform.
    #[serde(skip)]
    pub schema_locked: bool,
    /// Monotonically increasing version counter — incremented on every mutation.
    /// Used for optimistic concurrency control in transactions.
    #[serde(skip, default)]
    pub version: u64,
    /// Property key interner: maps InternedKey(u64) → original string.
    /// Populated during ingestion (add_nodes, CREATE, SET) and deserialization.
    /// Skipped during serde — rebuilt on load by the InternedKey Deserialize impl.
    #[serde(skip)]
    pub interner: StringInterner,
    /// Shared property schemas per node type: type_name → Arc<TypeSchema>.
    /// Populated during ingestion (add_nodes, CREATE) and compaction (load).
    #[serde(skip)]
    pub type_schemas: HashMap<String, Arc<TypeSchema>>,
    /// Fast-skip flag: true if any node has secondary labels.
    /// Read paths short-circuit the secondary_label_index scan entirely
    /// when this is false, so single-label graphs pay no perf tax.
    /// `#[serde(skip)]` — rebuilt by `rebuild_type_indices`.
    #[serde(skip)]
    pub has_secondary_labels: bool,
    /// O(1) secondary-label index: label_key → [NodeIndex].
    /// Populated by the choke-point label mutation API
    /// (`DirGraph::add_node_label` / `remove_node_label`) and on load by
    /// `rebuild_type_indices`. `#[serde(skip)]` — rebuilt from
    /// `NodeData.extra_labels` on load.
    #[serde(skip)]
    pub secondary_label_index: HashMap<InternedKey, Vec<NodeIndex>>,
}

pub(crate) fn default_auto_vacuum_threshold() -> Option<f64> {
    Some(0.3)
}

impl Drop for DirGraph {
    fn drop(&mut self) {
        // Clean up temp directories created during load or columnar spill.
        // Only the last Arc holder actually removes the dirs.
        if let Ok(dirs) = self.temp_dirs.lock() {
            // Only clean up if we're the sole owner (no other clones alive)
            if Arc::strong_count(&self.temp_dirs) <= 1 {
                for dir in dirs.iter() {
                    let _ = std::fs::remove_dir_all(dir);
                }
            }
        }
    }
}

impl Default for DirGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Warn (rate-limited, stderr) when building a type's id-index collapses
/// duplicate ids — `MATCH (n {id: …})` then returns only one node per id.
/// Detected here (at index build) rather than per-mutation so bulk
/// `UNWIND … CREATE` and `add_nodes` stay O(n), not O(n²). `id` is meant to
/// be unique (like `add_nodes(unique_id_field=…)`); use MERGE or dedupe input.
fn warn_on_duplicate_ids(node_type: &str, entry_count: usize, unique_count: usize) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static WARN_COUNT: AtomicUsize = AtomicUsize::new(0);
    if unique_count >= entry_count {
        return;
    }
    let dups = entry_count - unique_count;
    let seen = WARN_COUNT.fetch_add(1, Ordering::Relaxed);
    if seen < 5 {
        eprintln!(
            "warning: {dups} duplicate id(s) on type '{node_type}' — \
             `MATCH (n {{id: …}})` returns only one node per id. ids must be \
             unique; use MERGE or dedupe the input."
        );
    } else if seen == 5 {
        eprintln!("warning: further duplicate-id warnings suppressed.");
    }
}

impl DirGraph {
    /// Current monotonic version counter. Incremented on every
    /// mutation (via the kglite mutation paths). Used for optimistic
    /// concurrency control (OCC) by [`crate::graph::session`] and
    /// downstream consumers (the Python `Transaction` class, the
    /// `kglite-bolt-server` per-tx commit path).
    ///
    /// Exposed via `kglite::api::DirGraph::version` since Phase E;
    /// previously the field was `pub(crate)` only.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Set the version directly. Used by [`crate::graph::session::Session::commit`]
    /// to bump the working DirGraph's version on commit-swap. Not
    /// for general use — mutation paths bump version through their
    /// own mechanisms.
    pub fn set_version(&mut self, v: u64) {
        self.version = v;
    }

    pub fn new() -> Self {
        DirGraph {
            graph: GraphBackend::new(),
            type_indices: TypeIndexStore::new(),
            schema_definition: None,
            property_indices: HashMap::new(),
            composite_indices: HashMap::new(),
            property_index_keys: Vec::new(),
            composite_index_keys: Vec::new(),
            range_indices: HashMap::new(),
            range_index_keys: Vec::new(),
            id_indices: IdIndexStore::new(),
            connection_types: std::collections::HashSet::new(),
            node_type_metadata: HashMap::new(),
            connection_type_metadata: HashMap::new(),
            save_metadata: SaveMetadata::current(),
            id_field_aliases: FxHashMap::default(),
            title_field_aliases: FxHashMap::default(),
            parent_types: HashMap::new(),
            auto_vacuum_threshold: default_auto_vacuum_threshold(),
            spatial_configs: HashMap::new(),
            wkt_cache: Arc::new(RwLock::new(HashMap::new())),
            edge_type_counts_cache: Arc::new(RwLock::new(None)),
            type_connectivity_cache: Arc::new(RwLock::new(None)),
            embeddings: HashMap::new(),
            timeseries_configs: HashMap::new(),
            timeseries_store: HashMap::new(),
            temporal_node_configs: HashMap::new(),
            temporal_edge_configs: HashMap::new(),
            column_stores: HashMap::new(),
            memory_limit: None,
            spill_dir: None,
            temp_dirs: Arc::new(std::sync::Mutex::new(Vec::new())),
            read_only: false,
            schema_locked: false,
            version: 0,
            interner: StringInterner::new(),
            type_schemas: HashMap::new(),
            has_secondary_labels: false,
            secondary_label_index: HashMap::new(),
        }
    }

    /// Create a DirGraph from a pre-existing graph (used by v3 loader).
    /// All metadata fields start empty and are populated by the caller.
    pub fn from_graph(graph: GraphBackend) -> Self {
        DirGraph {
            graph,
            type_indices: TypeIndexStore::new(),
            schema_definition: None,
            property_indices: HashMap::new(),
            composite_indices: HashMap::new(),
            property_index_keys: Vec::new(),
            composite_index_keys: Vec::new(),
            range_indices: HashMap::new(),
            range_index_keys: Vec::new(),
            id_indices: IdIndexStore::new(),
            connection_types: std::collections::HashSet::new(),
            node_type_metadata: HashMap::new(),
            connection_type_metadata: HashMap::new(),
            save_metadata: SaveMetadata::default(),
            id_field_aliases: FxHashMap::default(),
            title_field_aliases: FxHashMap::default(),
            parent_types: HashMap::new(),
            auto_vacuum_threshold: default_auto_vacuum_threshold(),
            spatial_configs: HashMap::new(),
            wkt_cache: Arc::new(RwLock::new(HashMap::new())),
            edge_type_counts_cache: Arc::new(RwLock::new(None)),
            type_connectivity_cache: Arc::new(RwLock::new(None)),
            embeddings: HashMap::new(),
            timeseries_configs: HashMap::new(),
            timeseries_store: HashMap::new(),
            temporal_node_configs: HashMap::new(),
            temporal_edge_configs: HashMap::new(),
            column_stores: HashMap::new(),
            memory_limit: None,
            spill_dir: None,
            temp_dirs: Arc::new(std::sync::Mutex::new(Vec::new())),
            read_only: false,
            schema_locked: false,
            version: 0,
            interner: StringInterner::new(),
            type_schemas: HashMap::new(),
            has_secondary_labels: false,
            secondary_label_index: HashMap::new(),
        }
    }

    /// Look up spatial config for a node type.
    pub fn get_spatial_config(&self, node_type: &str) -> Option<&SpatialConfig> {
        self.spatial_configs.get(node_type)
    }

    /// Look up timeseries data for a specific node by its index.
    pub fn get_node_timeseries(
        &self,
        node_index: usize,
    ) -> Option<&crate::graph::features::timeseries::NodeTimeseries> {
        self.timeseries_store.get(&node_index)
    }

    /// Look up an embedding store by `(&str, &str)` without allocating owned Strings.
    /// Falls back to a linear scan of the embeddings map (typically 1-3 entries).
    #[inline]
    pub fn embedding_store(&self, node_type: &str, prop_name: &str) -> Option<&EmbeddingStore> {
        // Embedding maps are tiny (usually 1-5 entries), so linear scan beats allocation
        self.embeddings
            .iter()
            .find(|((nt, pn), _)| nt == node_type && pn == prop_name)
            .map(|(_, store)| store)
    }

    /// Build the ID index for a specific node type.
    /// Called lazily on first lookup for that type.
    pub fn build_id_index(&mut self, node_type: &str) {
        if self.id_indices.contains_key(node_type) {
            return; // Already built
        }
        let index = self.compute_id_index(node_type);
        self.id_indices.insert(node_type.to_string(), index);
    }

    /// Build id_index for a type using column stores directly (no node materialization).
    /// For DiskGraph, reads ids from mmap'd column stores via row_id from node_slots.
    /// Much faster and uses no arena memory. `compute_id_index` already
    /// prefers the column path for disk graphs, so this is a thin alias.
    pub fn build_id_index_from_columns(&mut self, node_type: &str) {
        self.build_id_index(node_type);
    }

    /// Compute (without inserting) the `TypeIdIndex` for a node type by
    /// scanning the graph. The shared body behind `build_id_index` (the
    /// &mut, cache-on-build path) and the read-path lazy build in
    /// `lookup_by_id_normalized` (via `IdIndexStore::lookup_or_build`, the
    /// &self self-healing path — see issue #20).
    ///
    /// Disk graphs with a column store read ids straight from the mmap'd
    /// columns (no node materialization); everything else scans node weights.
    fn compute_id_index(&self, node_type: &str) -> TypeIdIndex {
        let node_indices = match self.type_indices.get(node_type) {
            Some(indices) => indices,
            None => return TypeIdIndex::General(HashMap::new()),
        };

        let mut all_unique_id = true;
        let mut entries: Vec<(Value, NodeIndex)> = Vec::with_capacity(node_indices.len());

        // Disk + column store: read ids directly from mmap'd columns.
        let used_columns = if let GraphBackend::Disk(ref dg) = self.graph {
            if let Some(store) = self.column_stores.get(node_type) {
                for node_idx in node_indices.iter() {
                    let slot = dg.node_slot(node_idx.index());
                    if slot.is_alive() {
                        if let Some(id_val) = store.get_id(slot.row_id) {
                            if !matches!(id_val, Value::UniqueId(_)) {
                                all_unique_id = false;
                            }
                            entries.push((id_val, node_idx));
                        }
                    }
                }
                true
            } else {
                false
            }
        } else {
            false
        };

        // In-memory (and disk-without-column-store): scan node weights.
        if !used_columns {
            for node_idx in node_indices.iter() {
                if let Some(node) = self.graph.node_weight(node_idx) {
                    let node_id = node.id().into_owned();
                    if !matches!(node_id, Value::UniqueId(_)) {
                        all_unique_id = false;
                    }
                    entries.push((node_id, node_idx));
                }
            }
        }

        let entry_count = entries.len();
        if all_unique_id && !entries.is_empty() {
            // Compact: u32 keys only (~8 bytes per entry vs ~60).
            let map: HashMap<u32, NodeIndex> = entries
                .into_iter()
                .filter_map(|(id, idx)| {
                    if let Value::UniqueId(u) = id {
                        Some((u, idx))
                    } else {
                        None
                    }
                })
                .collect();
            warn_on_duplicate_ids(node_type, entry_count, map.len());
            TypeIdIndex::Integer(map)
        } else {
            // General: mixed ID types.
            let map: HashMap<Value, NodeIndex> = entries.into_iter().collect();
            warn_on_duplicate_ids(node_type, entry_count, map.len());
            TypeIdIndex::General(map)
        }
    }

    /// Look up a node by type and ID value. O(1) after index is built.
    /// Builds the index lazily if not already built.
    /// Handles type normalization: Python int may come as Int64 but be stored as UniqueId.
    pub fn lookup_by_id(&mut self, node_type: &str, id: &Value) -> Option<NodeIndex> {
        // The normalized path self-heals: it builds + caches the index on a
        // miss, so no separate build step is needed here.
        self.lookup_by_id_normalized(node_type, id)
    }

    /// Look up a node by type and ID value without building index.
    /// Use this for read-only access when index already exists.
    /// Handles type normalization for integer types.
    pub fn lookup_by_id_readonly(&self, node_type: &str, id: &Value) -> Option<NodeIndex> {
        self.lookup_by_id_normalized(node_type, id)
    }

    /// Lookup node by ID with automatic type normalization.
    /// This handles the Python-Rust type mismatch where Python int -> Int64 but
    /// DataFrame unique_id columns store as UniqueId(u32).
    ///
    /// O(1) self-healing: if the id_index for this type is missing (e.g. after
    /// `add_nodes` / `CREATE` / `DELETE` invalidated it), the index is built
    /// once on this read and cached in the overlay — every subsequent lookup
    /// is O(1). Replaces the old O(node-position) linear scan that re-ran on
    /// every `MATCH (n {id:X})` / `MERGE` match against an un-indexed type
    /// (issue #20). `TypeIdIndex::get` does the Int64↔UniqueId/Float/prefix
    /// normalization the old scan did by hand.
    pub fn lookup_by_id_normalized(&self, node_type: &str, id: &Value) -> Option<NodeIndex> {
        self.id_indices
            .lookup_or_build(node_type, id, || self.compute_id_index(node_type))
    }

    /// Set the schema definition for this graph
    pub fn set_schema(&mut self, schema: SchemaDefinition) {
        self.schema_definition = Some(schema);
    }

    /// Get the schema definition if one is set
    pub fn get_schema(&self) -> Option<&SchemaDefinition> {
        self.schema_definition.as_ref()
    }

    /// Clear the schema definition
    pub fn clear_schema(&mut self) {
        self.schema_definition = None;
    }

    pub fn has_connection_type(&self, connection_type: &str) -> bool {
        // Fast path: check the interned connection_types cache (O(1))
        if !self.connection_types.is_empty() {
            return self
                .connection_types
                .contains(&InternedKey::from_str(connection_type));
        }
        // Check metadata
        if self.connection_type_metadata.contains_key(connection_type) {
            return true;
        }
        // If metadata is empty (e.g. disk graph without full metadata),
        // check the interner — if the string was interned, it likely exists as
        // a connection type. This avoids false negatives that would cause
        // edge-type-filtered queries to return 0 results.
        if self.connection_type_metadata.is_empty() {
            return self
                .interner
                .try_resolve(InternedKey::from_str(connection_type))
                .is_some();
        }
        // Disk-side fall-through: even when the in-memory metadata
        // looks complete-but-stale (Cypher DETACH DELETE clears the
        // `connection_types` set but leaves `connection_type_metadata`
        // alone), the disk backend's `conn_type_index_*` mmap arrays
        // are authoritative for the live edge set. Asking the trait
        // for any source via the bounded helper is O(1) on disk —
        // returns `Some(non-empty)` if the conn type has at least
        // one live edge, `None` if no index for this name. 0.8.16.
        let key = InternedKey::from_str(connection_type);
        matches!(
            self.graph.sources_for_conn_type_bounded(key, Some(1)),
            Some(v) if !v.is_empty()
        )
    }

    /// Register a connection type (interned) for O(1) lookups.
    /// Called when edges are added to the graph.
    pub fn register_connection_type(&mut self, connection_type: String) {
        // If the cache has never been populated (disk-loaded graphs skip
        // `build_connection_types_cache` at load — only the v3 / file
        // loader calls it), backfill it from `connection_type_metadata`
        // before adding the new key. Otherwise the new key would land
        // in an empty set, flipping `has_connection_type` from "fall
        // through to metadata" mode (which sees every existing type) to
        // "use cache" mode (which returns false for every type except
        // this one). Manifested in 0.9.4 as: load disk graph →
        // add_connections of any new edge type → all subsequent
        // typed-anchored MATCH queries on existing edge types return 0
        // rows.
        if self.connection_types.is_empty() && !self.connection_type_metadata.is_empty() {
            self.build_connection_types_cache();
        }
        let key = self.interner.get_or_intern(&connection_type);
        self.connection_types.insert(key);
    }

    /// Build the connection types cache.
    /// Called after deserialization or when cache is needed.
    /// Fast path: populate from connection_type_metadata (O(types), no edge scan).
    /// Fallback: scan all edges (O(edges)) if metadata is empty.
    pub fn build_connection_types_cache(&mut self) {
        if !self.connection_types.is_empty() {
            return; // Already built
        }

        // Fast path: metadata is serialized — use it instead of scanning edges
        if !self.connection_type_metadata.is_empty() {
            for key in self.connection_type_metadata.keys() {
                self.connection_types
                    .insert(self.interner.get_or_intern(key));
            }
            return;
        }

        // Fallback: scan all edges (pre-metadata graphs)
        for edge in self.graph.edge_weights() {
            self.connection_types.insert(edge.connection_type);
        }
    }

    /// Compute edge counts grouped by connection type. Lazily cached.
    pub fn get_edge_type_counts(&self) -> HashMap<String, usize> {
        // Fast path: return cached result
        {
            let read = self.edge_type_counts_cache.read().unwrap();
            if let Some(ref cached) = *read {
                return cached.clone();
            }
        }
        // Slow path: compute O(E) and cache.
        // Uses edge_endpoint_keys() (mmap reads, zero heap per edge) instead of
        // edge_weights() (which materializes EdgeData → OOM on extreme-scale disk graphs).
        let mut counts: HashMap<InternedKey, usize> = HashMap::new();
        for (_src, _tgt, conn_key) in self.graph.edge_endpoint_keys() {
            *counts.entry(conn_key).or_insert(0) += 1;
        }
        // Resolve to strings
        let string_counts: HashMap<String, usize> = counts
            .into_iter()
            .map(|(k, v)| (self.interner.resolve(k).to_string(), v))
            .collect();
        let mut write = self.edge_type_counts_cache.write().unwrap();
        *write = Some(string_counts.clone());
        string_counts
    }

    /// Invalidate edge caches (call after edge mutations).
    pub(crate) fn invalidate_edge_type_counts_cache(&self) {
        *self.edge_type_counts_cache.write().unwrap() = None;
        *self.type_connectivity_cache.write().unwrap() = None;
    }

    /// Check if edge type count cache is populated (avoids O(E) scan).
    pub fn has_edge_type_counts_cache(&self) -> bool {
        self.edge_type_counts_cache.read().unwrap().is_some()
    }

    /// Check if type connectivity cache is populated.
    pub fn has_type_connectivity_cache(&self) -> bool {
        self.type_connectivity_cache.read().unwrap().is_some()
    }

    /// Get the type connectivity triples (if cached).
    pub fn get_type_connectivity(&self) -> Option<Vec<ConnectivityTriple>> {
        self.type_connectivity_cache.read().unwrap().clone()
    }

    /// Set the type connectivity cache.
    pub fn set_type_connectivity(&self, triples: Vec<ConnectivityTriple>) {
        *self.type_connectivity_cache.write().unwrap() = Some(triples);
    }

    /// Get (or compute) the label-pair edge-count triples — the
    /// `(src_type, edge_type, tgt_type) → count` cardinality cache
    /// used by the Cypher planner for selectivity-aware cost estimation.
    ///
    /// Lazy: on cold cache, walks every edge once via
    /// `edge_endpoint_keys()` and groups by `(src.node_type, conn_key,
    /// tgt.node_type)`. Identical shape to the n-triples loader's
    /// existing `set_type_connectivity(...)` output, so consumers can
    /// uniformly treat both as authoritative.
    ///
    /// On cache hit (common case after the first query), returns the
    /// cached `Vec` clone in O(triples) — typically <100 entries on
    /// real graphs, so essentially free.
    ///
    /// Invalidated alongside `edge_type_counts_cache` on every edge
    /// mutation.
    pub fn get_or_compute_type_connectivity(&self) -> Vec<ConnectivityTriple> {
        {
            let read = self.type_connectivity_cache.read().unwrap();
            if let Some(ref cached) = *read {
                return cached.clone();
            }
        }
        // Cold: O(E) walk grouping by (src_type, conn_type, tgt_type).
        let mut counts: HashMap<(InternedKey, InternedKey, InternedKey), usize> = HashMap::new();
        for (src_idx, tgt_idx, conn_key) in self.graph.edge_endpoint_keys() {
            let src_type = match self.graph.node_weight(src_idx) {
                Some(n) => n.node_type,
                None => continue,
            };
            let tgt_type = match self.graph.node_weight(tgt_idx) {
                Some(n) => n.node_type,
                None => continue,
            };
            *counts.entry((src_type, conn_key, tgt_type)).or_insert(0) += 1;
        }
        let triples: Vec<ConnectivityTriple> = counts
            .into_iter()
            .map(|((src, conn, tgt), count)| ConnectivityTriple {
                src: self.interner.resolve(src).to_string(),
                conn: self.interner.resolve(conn).to_string(),
                tgt: self.interner.resolve(tgt).to_string(),
                count,
            })
            .collect();
        *self.type_connectivity_cache.write().unwrap() = Some(triples.clone());
        triples
    }

    // ========================================================================
    // Type Metadata Methods (replaces SchemaNode graph nodes)
    // ========================================================================

    /// Get metadata for a node type (property names → type strings).
    pub fn get_node_type_metadata(&self, node_type: &str) -> Option<&HashMap<String, String>> {
        self.node_type_metadata.get(node_type)
    }

    /// Does any node type store a property named like a soft structural alias
    /// (`type` / `node_type` / `label`)? When true, `n.type` / `n.label` are
    /// property-first (KG-1) and no longer equal the node's primary type, so
    /// the `RETURN n.type, count(*)` count-fusion must NOT fire (it would
    /// group by the wrong key). `node_type_metadata` is the complete property
    /// catalogue — add_nodes and cypher CREATE both register into it and it
    /// round-trips through save/load — so this O(#types) plan-time scan is an
    /// exact gate. Cheap: only consulted for count-by-type-shaped queries.
    pub fn has_type_shadowing_property(&self) -> bool {
        self.node_type_metadata.values().any(|props| {
            props.contains_key("type")
                || props.contains_key("node_type")
                || props.contains_key("label")
        })
    }

    /// Upsert node type metadata — merges new property types into existing.
    pub fn upsert_node_type_metadata(&mut self, node_type: &str, props: HashMap<String, String>) {
        let entry = self
            .node_type_metadata
            .entry(node_type.to_string())
            .or_default();
        for (k, v) in props {
            entry.insert(k, v);
        }
    }

    /// Upsert connection type metadata — merges property types and accumulates type pairs.
    pub fn upsert_connection_type_metadata(
        &mut self,
        conn_type: &str,
        source_type: &str,
        target_type: &str,
        prop_types: HashMap<String, String>,
    ) {
        let entry = self
            .connection_type_metadata
            .entry(conn_type.to_string())
            .or_insert_with(|| ConnectionTypeInfo {
                source_types: HashSet::new(),
                target_types: HashSet::new(),
                property_types: HashMap::new(),
            });
        entry.source_types.insert(source_type.to_string());
        entry.target_types.insert(target_type.to_string());
        for (k, v) in prop_types {
            entry.property_types.insert(k, v);
        }
    }

    pub fn has_node_type(&self, node_type: &str) -> bool {
        self.type_indices.contains_key(node_type) || self.node_type_metadata.contains_key(node_type)
    }

    /// Get all node types that exist in the graph.
    pub fn get_node_types(&self) -> Vec<String> {
        let mut types: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Get types from type_indices
        for node_type in self.type_indices.keys() {
            types.insert(node_type.to_string());
        }

        // Also include types from metadata (may have metadata but no live nodes)
        for node_type in self.node_type_metadata.keys() {
            types.insert(node_type.clone());
        }

        types.into_iter().collect()
    }

    /// Resolve a property name through field aliases.
    /// If the property matches the original ID or title field name for this node type,
    /// returns the canonical name ("id" or "title"). Otherwise returns the property unchanged.
    pub fn resolve_alias<'a>(&'a self, node_type: &str, property: &'a str) -> &'a str {
        if self.id_field_aliases.is_empty() && self.title_field_aliases.is_empty() {
            return property;
        }
        if let Some(alias) = self.id_field_aliases.get(node_type) {
            if alias == property {
                return "id";
            }
        }
        if let Some(alias) = self.title_field_aliases.get(node_type) {
            if alias == property {
                return "title";
            }
        }
        property
    }

    pub fn get_node(&self, index: NodeIndex) -> Option<&NodeData> {
        self.graph.node_weight(index)
    }

    pub fn get_node_mut(&mut self, index: NodeIndex) -> Option<&mut NodeData> {
        self.graph.node_weight_mut(index)
    }

    pub fn _get_connection(&self, index: EdgeIndex) -> Option<&EdgeData> {
        self.graph.edge_weight(index)
    }

    pub fn _get_connection_mut(&mut self, index: EdgeIndex) -> Option<&mut EdgeData> {
        self.graph.edge_weight_mut(index)
    }

    // ========================================================================
    // Index Management Methods
    // ========================================================================

    /// Create an index on a property for a specific node type.
    /// Returns the number of entries indexed.
    ///
    /// The id-alias / title-alias fields (e.g. `add_nodes(df, "Star",
    /// "starId", "title")` makes `starId` the alias for the canonical
    /// id) are intentionally NOT special-cased here: their indices
    /// would build as empty (id/title live off the properties map).
    /// Lookups against id-alias names route through `lookup_by_id_readonly`
    /// in the matcher (`try_index_lookup`), which uses the auto-
    /// maintained per-type `id_index` — no separate `create_index` call
    /// required, and SET-on-id always stays in sync because id mutation
    /// updates the id_index directly.
    pub fn create_index(&mut self, node_type: &str, property: &str) -> usize {
        // Store key uses the user's `property` name verbatim — the
        // matcher's `try_index_lookup` indexes into `property_indices`
        // by the unresolved user-facing key (matcher.rs:850), so the
        // auto-maintenance path keeps things in sync only when the
        // storage key matches.
        let store_key = (node_type.to_string(), property.to_string());

        // Mirror the matcher's property-READ path
        // (`core/pattern_matching/matcher.rs::
        // node_matches_properties_columnar`) so the index covers the
        // same value-space MATCH consults. Three concerns the read
        // path handles that `node.get_property()` did not:
        //
        // 1. **Alias resolution.** `starId` may be an id-alias for
        //    `id`; same for title-aliases. Without resolving, we'd
        //    look up "starId" in PropertyStorage and miss the data
        //    (stored under "id").
        // 2. **`id` / `title` are special.** Their values live in
        //    `node_slots` (disk) / dedicated NodeData fields, NOT in
        //    `properties`. The matcher reads them via `get_node_id`
        //    / `get_node_title`; we do the same.
        // 3. **Column-aware reads.** For mapped/disk graphs loaded
        //    from .kgl, the actual values live in a `ColumnStore`,
        //    not in the node's `PropertyStorage::Map`/`Compact`
        //    snapshot. The backend's `get_node_property` knows how
        //    to read from each storage type; `NodeData::get_property`
        //    only reads the in-memory snapshot and silently returns
        //    None for column-stored values.
        let read_key = self.resolve_alias(node_type, property).to_string();
        let interned_key = self.interner.get_or_intern(&read_key);
        let mut index: HashMap<Value, Vec<NodeIndex>> = HashMap::new();

        if let Some(node_indices) = self.type_indices.get(node_type) {
            for idx in node_indices.iter() {
                let value = if read_key == "id" {
                    self.graph.get_node_id(idx)
                } else if read_key == "title" {
                    self.graph.get_node_title(idx)
                } else {
                    self.graph.get_node_property(idx, interned_key)
                };
                if let Some(value) = value {
                    index.entry(value).or_default().push(idx);
                }
            }
        }

        let count = index.len();
        self.property_indices.insert(store_key, index);
        count
    }

    /// Drop an index on a property for a specific node type.
    /// Returns true if the index existed and was removed.
    pub fn drop_index(&mut self, node_type: &str, property: &str) -> bool {
        let key = (node_type.to_string(), property.to_string());
        self.property_indices.remove(&key).is_some()
    }

    /// Check if an index exists for a given node type and property.
    pub fn has_index(&self, node_type: &str, property: &str) -> bool {
        let key = (node_type.to_string(), property.to_string());
        self.property_indices.contains_key(&key)
    }

    /// Check if **any** index exists for `(node_type, property)` — the
    /// in-memory `property_indices` HashMap *or* a persistent
    /// disk-backed `PropertyIndex`. Used by `describe()` to annotate
    /// schema output with `indexed=…` attributes so agents can tell
    /// which properties hit an O(log N) path.
    pub fn has_any_index(&self, node_type: &str, property: &str) -> bool {
        if self.has_index(node_type, property) {
            return true;
        }
        if let crate::graph::storage::backend::GraphBackend::Disk(dg) = &self.graph {
            return dg.has_property_index(node_type, property);
        }
        false
    }

    /// Get all existing indexes as a list of (node_type, property) tuples.
    pub fn list_indexes(&self) -> Vec<(String, String)> {
        self.property_indices.keys().cloned().collect()
    }

    /// Look up nodes by property value using an index.
    /// Returns None if no index exists, otherwise returns matching node indices.
    pub fn lookup_by_index(
        &self,
        node_type: &str,
        property: &str,
        value: &Value,
    ) -> Option<Vec<NodeIndex>> {
        let key = (node_type.to_string(), property.to_string());
        self.property_indices
            .get(&key)
            .and_then(|idx| idx.get(value))
            .cloned()
    }

    /// Get statistics about an index.
    pub fn get_index_stats(&self, node_type: &str, property: &str) -> Option<IndexStats> {
        let key = (node_type.to_string(), property.to_string());
        self.property_indices.get(&key).map(|idx| {
            let total_entries: usize = idx.values().map(|v| v.len()).sum();
            IndexStats {
                unique_values: idx.len(),
                total_entries,
                avg_entries_per_value: if idx.is_empty() {
                    0.0
                } else {
                    total_entries as f64 / idx.len() as f64
                },
            }
        })
    }

    // ========================================================================
    // Range Index Methods (B-Tree)
    // ========================================================================

    /// Create a range index (B-Tree) on a property for a specific node type.
    /// Enables efficient range queries (>, >=, <, <=, BETWEEN).
    /// Returns the number of unique values indexed.
    pub fn create_range_index(&mut self, node_type: &str, property: &str) -> usize {
        let key = (node_type.to_string(), property.to_string());
        // Same alias-resolution + column-aware property read as
        // `create_index` — see the comment there for the full
        // rationale on why we don't use `node.get_property`.
        let resolved = self.resolve_alias(node_type, property).to_string();
        let interned_key = self.interner.get_or_intern(&resolved);
        let mut index: std::collections::BTreeMap<Value, Vec<NodeIndex>> =
            std::collections::BTreeMap::new();

        if let Some(node_indices) = self.type_indices.get(node_type) {
            for idx in node_indices.iter() {
                let value = if resolved == "id" {
                    self.graph.get_node_id(idx)
                } else if resolved == "title" {
                    self.graph.get_node_title(idx)
                } else {
                    self.graph.get_node_property(idx, interned_key)
                };
                if let Some(value) = value {
                    index.entry(value).or_default().push(idx);
                }
            }
        }

        let count = index.len();
        self.range_indices.insert(key, index);
        count
    }

    /// Drop a range index. Returns true if it existed.
    pub fn drop_range_index(&mut self, node_type: &str, property: &str) -> bool {
        let key = (node_type.to_string(), property.to_string());
        self.range_indices.remove(&key).is_some()
    }

    /// Range lookup: returns node indices where property value falls in the given range.
    pub fn lookup_range(
        &self,
        node_type: &str,
        property: &str,
        lower: std::ops::Bound<&Value>,
        upper: std::ops::Bound<&Value>,
    ) -> Option<Vec<NodeIndex>> {
        let key = (node_type.to_string(), property.to_string());
        self.range_indices.get(&key).map(|btree| {
            btree
                .range((lower, upper))
                .flat_map(|(_, indices)| indices.iter().copied())
                .collect()
        })
    }

    // ========================================================================
    // Composite Index Methods
    // ========================================================================

    /// Create a composite index on multiple properties for a specific node type.
    /// Composite indexes enable efficient lookups on multiple fields at once.
    ///
    /// Returns the number of unique value combinations indexed.
    ///
    /// Example: create_composite_index("Person", &["city", "age"]) allows efficient
    /// queries like filter({'city': 'Oslo', 'age': 30}).
    pub fn create_composite_index(&mut self, node_type: &str, properties: &[&str]) -> usize {
        let key = (
            node_type.to_string(),
            properties.iter().map(|s| s.to_string()).collect(),
        );

        // Pre-resolve each property name (alias → canonical "id" /
        // "title" when applicable) and pre-intern so the per-node
        // loop is HashMap-only. Mirrors the matcher's read path —
        // see `create_index`'s comment for the full rationale.
        let resolved: Vec<String> = properties
            .iter()
            .map(|p| self.resolve_alias(node_type, p).to_string())
            .collect();
        let interned_keys: Vec<InternedKey> = resolved
            .iter()
            .map(|r| self.interner.get_or_intern(r))
            .collect();

        let mut index: HashMap<CompositeValue, Vec<NodeIndex>> = HashMap::new();

        if let Some(node_indices) = self.type_indices.get(node_type) {
            for idx in node_indices.iter() {
                let values: Vec<Value> = resolved
                    .iter()
                    .zip(interned_keys.iter())
                    .map(|(r, k)| {
                        let v = if r == "id" {
                            self.graph.get_node_id(idx)
                        } else if r == "title" {
                            self.graph.get_node_title(idx)
                        } else {
                            self.graph.get_node_property(idx, *k)
                        };
                        v.unwrap_or(Value::Null)
                    })
                    .collect();

                // Only index if at least one value is non-null
                if values.iter().any(|v| !matches!(v, Value::Null)) {
                    index.entry(CompositeValue(values)).or_default().push(idx);
                }
            }
        }

        let count = index.len();
        self.composite_indices.insert(key, index);
        count
    }

    /// Drop a composite index.
    /// Returns true if the index existed and was removed.
    pub fn drop_composite_index(&mut self, node_type: &str, properties: &[String]) -> bool {
        let key = (node_type.to_string(), properties.to_vec());
        self.composite_indices.remove(&key).is_some()
    }

    /// Check if a composite index exists.
    pub fn has_composite_index(&self, node_type: &str, properties: &[String]) -> bool {
        let key = (node_type.to_string(), properties.to_vec());
        self.composite_indices.contains_key(&key)
    }

    /// Get all existing composite indexes.
    pub fn list_composite_indexes(&self) -> Vec<(String, Vec<String>)> {
        self.composite_indices.keys().cloned().collect()
    }

    /// Look up nodes by composite values using a composite index.
    /// Properties must match the order used when creating the index.
    pub fn lookup_by_composite_index(
        &self,
        node_type: &str,
        properties: &[String],
        values: &[Value],
    ) -> Option<Vec<NodeIndex>> {
        let key = (node_type.to_string(), properties.to_vec());
        let composite_value = CompositeValue(values.to_vec());

        self.composite_indices
            .get(&key)
            .and_then(|idx| idx.get(&composite_value))
            .cloned()
    }

    /// Get statistics about a composite index.
    pub fn get_composite_index_stats(
        &self,
        node_type: &str,
        properties: &[String],
    ) -> Option<IndexStats> {
        let key = (node_type.to_string(), properties.to_vec());
        self.composite_indices.get(&key).map(|idx| {
            let total_entries: usize = idx.values().map(|v| v.len()).sum();
            IndexStats {
                unique_values: idx.len(),
                total_entries,
                avg_entries_per_value: if idx.is_empty() {
                    0.0
                } else {
                    total_entries as f64 / idx.len() as f64
                },
            }
        })
    }

    /// Find a composite index that can be used for a given set of filter properties.
    /// Returns the index key and whether all filter properties are covered.
    pub fn find_matching_composite_index(
        &self,
        node_type: &str,
        filter_properties: &[String],
    ) -> Option<(CompositeIndexKey, bool)> {
        // Sort filter properties for comparison
        let mut sorted_filter: Vec<String> = filter_properties.to_vec();
        sorted_filter.sort();

        for key in self.composite_indices.keys() {
            if key.0 == node_type {
                let mut sorted_index: Vec<String> = key.1.clone();
                sorted_index.sort();

                // Check if index properties are a subset of or equal to filter properties
                // For exact match, the index must cover exactly the filter fields
                if sorted_index == sorted_filter {
                    return Some((key.clone(), true)); // Exact match
                }

                // Check if index is a prefix of filter (can be used for partial filtering)
                if sorted_filter.starts_with(&sorted_index)
                    || sorted_index.iter().all(|p| sorted_filter.contains(p))
                {
                    return Some((key.clone(), false)); // Partial match
                }
            }
        }
        None
    }

    // ========================================================================
    // Incremental Index Maintenance (called by Cypher mutations)
    // ========================================================================

    /// Update property, composite, and range indices after a new node is added.
    /// Only updates indices that already exist for this node_type.
    pub fn update_property_indices_for_add(&mut self, node_type: &str, node_idx: NodeIndex) {
        // Collect single-property index updates (immutable borrow of self.graph)
        let prop_updates: Vec<(IndexKey, Value)> = {
            let node = match self.graph.node_weight(node_idx) {
                Some(n) => n,
                None => return,
            };
            self.property_indices
                .keys()
                .chain(self.range_indices.keys())
                .filter(|(nt, _)| nt == node_type)
                .filter_map(|key| {
                    node.get_property(&key.1)
                        .map(|v| (key.clone(), v.into_owned()))
                })
                .collect()
        };
        for (key, value) in &prop_updates {
            if let Some(value_map) = self.property_indices.get_mut(key) {
                value_map.entry(value.clone()).or_default().push(node_idx);
            }
            if let Some(btree) = self.range_indices.get_mut(key) {
                btree.entry(value.clone()).or_default().push(node_idx);
            }
        }

        // Collect composite index updates
        let comp_updates: Vec<(CompositeIndexKey, CompositeValue)> = {
            let node = match self.graph.node_weight(node_idx) {
                Some(n) => n,
                None => return,
            };
            self.composite_indices
                .keys()
                .filter(|(nt, _)| nt == node_type)
                .filter_map(|key| {
                    let vals: Vec<Value> = key
                        .1
                        .iter()
                        .map(|p| {
                            node.get_property(p)
                                .map(Cow::into_owned)
                                .unwrap_or(Value::Null)
                        })
                        .collect();
                    if vals.iter().any(|v| !matches!(v, Value::Null)) {
                        Some((key.clone(), CompositeValue(vals)))
                    } else {
                        None
                    }
                })
                .collect()
        };
        for (key, comp_val) in comp_updates {
            if let Some(comp_map) = self.composite_indices.get_mut(&key) {
                comp_map.entry(comp_val).or_default().push(node_idx);
            }
        }
    }

    /// Update property, range, and composite indices after a property value is changed.
    /// Removes node from the old value bucket and adds to the new value bucket.
    pub fn update_property_indices_for_set(
        &mut self,
        node_type: &str,
        node_idx: NodeIndex,
        property: &str,
        old_value: Option<&Value>,
        new_value: &Value,
    ) {
        let key = (node_type.to_string(), property.to_string());
        // Update hash index
        if let Some(value_map) = self.property_indices.get_mut(&key) {
            if let Some(old_val) = old_value {
                if let Some(indices) = value_map.get_mut(old_val) {
                    indices.retain(|&idx| idx != node_idx);
                    if indices.is_empty() {
                        value_map.remove(old_val);
                    }
                }
            }
            value_map
                .entry(new_value.clone())
                .or_default()
                .push(node_idx);
        }
        // Update range index
        if let Some(btree) = self.range_indices.get_mut(&key) {
            if let Some(old_val) = old_value {
                if let Some(indices) = btree.get_mut(old_val) {
                    indices.retain(|&idx| idx != node_idx);
                    if indices.is_empty() {
                        btree.remove(old_val);
                    }
                }
            }
            btree.entry(new_value.clone()).or_default().push(node_idx);
        }

        // Update any composite indices that include this property
        self.update_composite_indices_for_property_change(node_type, node_idx, property);
    }

    /// Update property, range, and composite indices after a property is removed.
    pub fn update_property_indices_for_remove(
        &mut self,
        node_type: &str,
        node_idx: NodeIndex,
        property: &str,
        old_value: &Value,
    ) {
        let key = (node_type.to_string(), property.to_string());
        if let Some(value_map) = self.property_indices.get_mut(&key) {
            if let Some(indices) = value_map.get_mut(old_value) {
                indices.retain(|&idx| idx != node_idx);
                if indices.is_empty() {
                    value_map.remove(old_value);
                }
            }
        }
        if let Some(btree) = self.range_indices.get_mut(&key) {
            if let Some(indices) = btree.get_mut(old_value) {
                indices.retain(|&idx| idx != node_idx);
                if indices.is_empty() {
                    btree.remove(old_value);
                }
            }
        }

        // Update any composite indices that include this property
        self.update_composite_indices_for_property_change(node_type, node_idx, property);
    }

    /// Re-index a single node in all composite indices that include the changed property.
    /// Reads current node properties to build the new composite value.
    fn update_composite_indices_for_property_change(
        &mut self,
        node_type: &str,
        node_idx: NodeIndex,
        changed_property: &str,
    ) {
        let comp_keys: Vec<CompositeIndexKey> = self
            .composite_indices
            .keys()
            .filter(|(nt, props)| nt == node_type && props.contains(&changed_property.to_string()))
            .cloned()
            .collect();

        if comp_keys.is_empty() {
            return;
        }

        // Read current node properties once
        let current_props: HashMap<String, Value> = match self.graph.node_weight(node_idx) {
            Some(node) => node.properties_cloned(&self.interner),
            None => return,
        };

        for key in comp_keys {
            if let Some(comp_map) = self.composite_indices.get_mut(&key) {
                // Remove node from all existing composite buckets
                for indices in comp_map.values_mut() {
                    indices.retain(|&idx| idx != node_idx);
                }
                // Remove empty buckets
                comp_map.retain(|_, v| !v.is_empty());

                // Build new composite value from current properties
                let new_values: Vec<Value> = key
                    .1
                    .iter()
                    .map(|p| current_props.get(p).cloned().unwrap_or(Value::Null))
                    .collect();
                if new_values.iter().any(|v| !matches!(v, Value::Null)) {
                    comp_map
                        .entry(CompositeValue(new_values))
                        .or_default()
                        .push(node_idx);
                }
            }
        }
    }

    // ========================================================================
    // Serialization helpers
    // ========================================================================

    /// Snapshot which property/composite indexes exist so they survive serialization.
    /// Called automatically before save.
    /// Sync node_type_metadata to match actual column store contents.
    /// Removes properties from metadata that have no data in any column store.
    /// Called before save to ensure metadata consistency.
    pub fn populate_index_keys(&mut self) {
        self.property_index_keys = self.property_indices.keys().cloned().collect();
        self.composite_index_keys = self.composite_indices.keys().cloned().collect();
        self.range_index_keys = self.range_indices.keys().cloned().collect();
    }

    /// Rebuild property and composite indexes from the persisted key lists.
    /// Called automatically after load.
    pub fn rebuild_indices_from_keys(&mut self) {
        let prop_keys: Vec<IndexKey> = std::mem::take(&mut self.property_index_keys);
        for (node_type, property) in &prop_keys {
            self.create_index(node_type, property);
        }
        self.property_index_keys = prop_keys;

        let comp_keys: Vec<CompositeIndexKey> = std::mem::take(&mut self.composite_index_keys);
        for (node_type, properties) in &comp_keys {
            let prop_refs: Vec<&str> = properties.iter().map(|s| s.as_str()).collect();
            self.create_composite_index(node_type, &prop_refs);
        }
        self.composite_index_keys = comp_keys;

        let range_keys: Vec<IndexKey> = std::mem::take(&mut self.range_index_keys);
        for (node_type, property) in &range_keys {
            self.create_range_index(node_type, property);
        }
        self.range_index_keys = range_keys;
    }

    // ========================================================================
    // Graph Maintenance: reindex, vacuum, graph_info
    // ========================================================================

    /// Rebuild all indexes from the current graph state.
    ///
    /// Reconstructs type_indices, property_indices, and composite_indices by
    /// scanning all live nodes. Clears lazy caches (id_indices, connection_types)
    /// so they rebuild on next access.
    ///
    /// Use after bulk mutations to ensure index consistency, or when you suspect
    /// indexes have drifted from the actual graph state.
    /// Rebuild type_indices from the live graph.
    /// Called after deserialization (type_indices is `#[serde(skip)]`) and by `reindex()`.
    pub fn rebuild_type_indices(&mut self) {
        let type_count = self.node_type_metadata.len().max(4);
        let avg_per_type = self.graph.node_count() / type_count.max(1);
        let mut new_type_indices: HashMap<String, Vec<NodeIndex>> =
            HashMap::with_capacity(type_count);
        for node_idx in self.graph.node_indices() {
            if let Some(node) = self.graph.node_weight(node_idx) {
                let type_str = node.node_type_str(&self.interner).to_string();
                new_type_indices
                    .entry(type_str)
                    .or_insert_with(|| Vec::with_capacity(avg_per_type))
                    .push(node_idx);
            }
        }
        self.type_indices.replace_with(new_type_indices);
        // `secondary_label_index` is *not* rebuilt from node data — it's
        // the canonical store, populated either by the choke-point API
        // during the session or by the load path (the disk sidecar /
        // the in-memory .kgl section).
    }

    /// Add a secondary label to a node. Choke-point API for label
    /// mutations — every mutation site routes through here so the
    /// `secondary_label_index` stays canonical across all three
    /// backends. NodeData itself never carries extra labels; the
    /// inverted index is the single source of truth.
    ///
    /// Returns `true` if the label was added, `false` if it was already
    /// present (idempotent) or equal to the primary type.
    pub fn add_node_label(&mut self, idx: NodeIndex, label: InternedKey) -> bool {
        use crate::graph::storage::GraphRead;
        let primary = match GraphRead::node_type_of(&self.graph, idx) {
            Some(k) => k,
            None => return false,
        };
        if primary == label {
            return false;
        }
        let bucket = self.secondary_label_index.entry(label).or_default();
        if bucket.contains(&idx) {
            return false;
        }
        bucket.push(idx);
        self.has_secondary_labels = true;
        true
    }

    /// Remove a secondary label from a node. Choke-point API for label
    /// mutations.
    ///
    /// Returns `Ok(true)` if removed, `Ok(false)` if the node never had
    /// the label, `Err(...)` if `label` is the primary type (use
    /// `SET n.type = ...` to retype instead).
    pub fn remove_node_label(
        &mut self,
        idx: NodeIndex,
        label: InternedKey,
    ) -> Result<bool, String> {
        use crate::graph::storage::GraphRead;
        let Some(primary) = GraphRead::node_type_of(&self.graph, idx) else {
            return Ok(false);
        };
        if primary == label {
            return Err(
                "Cannot remove a node's primary label via REMOVE n:Label; use \
                 SET n.type = 'NewType' to retype."
                    .to_string(),
            );
        }
        let Some(bucket) = self.secondary_label_index.get_mut(&label) else {
            return Ok(false);
        };
        let before = bucket.len();
        bucket.retain(|&i| i != idx);
        let removed = bucket.len() < before;
        if removed && bucket.is_empty() {
            self.secondary_label_index.remove(&label);
        }
        if self.secondary_label_index.is_empty() {
            self.has_secondary_labels = false;
        }
        Ok(removed)
    }

    /// Return a node's labels as `[primary, ...extras]`. Returns an
    /// empty Vec if the node is missing. Consumers that only need the
    /// primary type should keep using `GraphRead::node_type_of` (one
    /// InternedKey lookup, no allocation).
    ///
    /// Reads secondaries from `secondary_label_index` (the canonical
    /// source maintained by the choke-point API) rather than from
    /// `NodeData.extra_labels`. This works uniformly across backends:
    /// in-memory + mapped have both in sync; on disk, `NodeData` is
    /// materialised from a transient arena that doesn't carry the
    /// extras, but the inverted index does.
    pub fn node_labels(&self, idx: NodeIndex) -> Vec<InternedKey> {
        use crate::graph::storage::GraphRead;
        let Some(primary) = GraphRead::node_type_of(&self.graph, idx) else {
            return Vec::new();
        };
        if !self.has_secondary_labels {
            return vec![primary];
        }
        let mut labels = vec![primary];
        for (&key, bucket) in &self.secondary_label_index {
            if bucket.contains(&idx) {
                labels.push(key);
            }
        }
        labels
    }

    /// All nodes carrying `label` as EITHER their primary type or a
    /// secondary label — the canonical "candidates for `MATCH (n:label)`"
    /// set. This is the single source of truth that every label-based
    /// candidate-selection site should route through, mirroring
    /// `PatternExecutor::find_matching_nodes`'s `needs_secondary_path`.
    ///
    /// Single-label fast path: when no node anywhere carries a secondary
    /// label, this returns exactly `type_indices[label].to_vec()` — byte
    /// for byte what every primary-only call site produced before
    /// multi-label existed, so single-label performance is unchanged.
    ///
    /// The choke-point API (`add_node_label`) forbids a node holding the
    /// same key as both primary and secondary, so the union is
    /// duplicate-free.
    pub fn nodes_with_label(&self, label: &str) -> Vec<NodeIndex> {
        let mut out = self
            .type_indices
            .get(label)
            .map(|v| v.to_vec())
            .unwrap_or_default();
        if self.has_secondary_labels {
            if let Some(secondary) = self
                .secondary_label_index
                .get(&InternedKey::from_str(label))
            {
                out.extend(secondary.iter().copied());
            }
        }
        out
    }

    /// True if `idx` carries `key` as its primary type or a secondary
    /// label. Membership test companion to `nodes_with_label` for sites
    /// that filter an existing candidate set rather than enumerate one.
    pub fn node_has_label(&self, idx: NodeIndex, key: InternedKey) -> bool {
        use crate::graph::storage::GraphRead;
        if GraphRead::node_type_of(&self.graph, idx) == Some(key) {
            return true;
        }
        self.has_secondary_labels
            && self
                .secondary_label_index
                .get(&key)
                .is_some_and(|bucket| bucket.contains(&idx))
    }

    /// Convert all node properties from PropertyStorage::Map to PropertyStorage::Compact.
    /// Called after deserialization to convert the transient Map storage to dense slot-vec.
    /// Builds TypeSchemas per node type and stores them in `self.type_schemas`.
    pub fn compact_properties(&mut self) {
        // Phase 1: Build TypeSchemas from node_type_metadata (O(types), not O(N×P))
        let mut schemas: HashMap<String, TypeSchema> = HashMap::new();
        for (node_type, props) in &self.node_type_metadata {
            let keys = props.keys().map(|name| self.interner.get_or_intern(name));
            schemas.insert(node_type.clone(), TypeSchema::from_keys(keys));
        }

        // Fallback: if metadata is empty (pre-metadata graph), scan nodes
        if schemas.is_empty() {
            for node_idx in self.graph.node_indices() {
                if let Some(node) = self.graph.node_weight(node_idx) {
                    let type_str = node.node_type_str(&self.interner).to_string();
                    let schema = schemas.entry(type_str).or_insert_with(TypeSchema::new);
                    if let PropertyStorage::Map(map) = &node.properties {
                        for &key in map.keys() {
                            schema.add_key(key);
                        }
                    }
                }
            }
        }

        // Phase 2: Wrap in Arc and store
        let arc_schemas: HashMap<String, Arc<TypeSchema>> =
            schemas.into_iter().map(|(t, s)| (t, Arc::new(s))).collect();

        // Phase 3: Convert each node's Map → Compact
        // Collect indices first to avoid borrowing conflict.
        let node_indices: Vec<NodeIndex> = self.graph.node_indices().collect();
        for node_idx in node_indices {
            let node = self.graph.node_weight_mut(node_idx).unwrap();
            if let PropertyStorage::Map(_) = &node.properties {
                let type_str = node.node_type_str(&self.interner);
                if let Some(schema) = arc_schemas.get(type_str) {
                    let old = std::mem::replace(
                        &mut node.properties,
                        PropertyStorage::Compact {
                            schema: Arc::clone(schema),
                            values: Vec::new(),
                        },
                    );
                    if let PropertyStorage::Map(map) = old {
                        node.properties = PropertyStorage::from_compact(map, schema);
                    }
                }
            }
        }

        self.type_schemas = arc_schemas;
    }

    /// Combined rebuild_type_indices + compact_properties in a single pass.
    /// Used after deserialization when both need to run.
    pub fn rebuild_type_indices_and_compact(&mut self) {
        // Build TypeSchemas from metadata (O(types))
        let mut schemas: HashMap<String, TypeSchema> = HashMap::new();
        for (node_type, props) in &self.node_type_metadata {
            let keys = props.keys().map(|name| self.interner.get_or_intern(name));
            schemas.insert(node_type.clone(), TypeSchema::from_keys(keys));
        }

        // Fallback: if metadata is empty (loaded from file), scan nodes
        if schemas.is_empty() {
            for node_idx in self.graph.node_indices() {
                if let Some(node) = self.graph.node_weight(node_idx) {
                    let type_str = node.node_type_str(&self.interner).to_string();
                    let schema = schemas.entry(type_str).or_insert_with(TypeSchema::new);
                    if let PropertyStorage::Map(map) = &node.properties {
                        for &key in map.keys() {
                            schema.add_key(key);
                        }
                    }
                }
            }
        }

        let arc_schemas: HashMap<String, Arc<TypeSchema>> =
            schemas.into_iter().map(|(t, s)| (t, Arc::new(s))).collect();

        // Single pass: build type_indices AND convert Map → Compact
        let type_count = arc_schemas.len().max(4);
        let avg_per_type = self.graph.node_count() / type_count.max(1);
        let mut new_type_indices: HashMap<String, Vec<NodeIndex>> =
            HashMap::with_capacity(type_count);

        let node_indices: Vec<NodeIndex> = self.graph.node_indices().collect();
        for node_idx in node_indices {
            let node = self.graph.node_weight_mut(node_idx).unwrap();

            // Rebuild type_indices
            let type_str = node.node_type_str(&self.interner).to_string();
            new_type_indices
                .entry(type_str)
                .or_insert_with(|| Vec::with_capacity(avg_per_type))
                .push(node_idx);

            // Convert Map → Compact
            if let PropertyStorage::Map(_) = &node.properties {
                let type_str = node.node_type_str(&self.interner);
                if let Some(schema) = arc_schemas.get(type_str) {
                    let old = std::mem::replace(
                        &mut node.properties,
                        PropertyStorage::Compact {
                            schema: Arc::clone(schema),
                            values: Vec::new(),
                        },
                    );
                    if let PropertyStorage::Map(map) = old {
                        node.properties = PropertyStorage::from_compact(map, schema);
                    }
                }
            }
        }

        self.type_indices.replace_with(new_type_indices);
        self.type_schemas = arc_schemas;
        // `secondary_label_index` is *not* rebuilt here — it's the
        // canonical store, populated by the load path (the disk
        // sidecar or the in-memory `.kgl` section).
    }

    /// Convert the graph to disk-backed storage mode.
    /// Enables columnar storage first, then builds CSR edge arrays on disk.
    /// Nodes stay in memory (~40 bytes each), edges are mmap'd.
    pub fn enable_disk_mode(&mut self) -> Result<(), String> {
        // Ensure columnar storage for compact node representation
        if !self.is_columnar() {
            self.enable_columnar();
        }

        // Create a temp directory for CSR files
        let data_dir = std::env::temp_dir().join(format!(
            "kglite_disk_{}_{:x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));

        // Extract the StableDiGraph and build DiskGraph
        let disk_graph = match &mut self.graph {
            GraphBackend::Memory(g) => {
                crate::graph::storage::disk::graph::DiskGraph::from_stable_digraph(
                    g.inner_mut(),
                    &data_dir,
                )
            }
            GraphBackend::Mapped(g) => {
                crate::graph::storage::disk::graph::DiskGraph::from_stable_digraph(
                    g.inner_mut(),
                    &data_dir,
                )
            }
            GraphBackend::Disk(_) => return Err("Already in disk mode".to_string()),
            GraphBackend::Recording(_) => {
                return Err(
                    "enable_disk_mode not supported while wrapped in RecordingGraph".to_string(),
                )
            }
        }
        .map_err(|e| format!("Failed to create DiskGraph: {}", e))?;

        // Register temp dir for cleanup
        if let Ok(mut dirs) = self.temp_dirs.lock() {
            dirs.push(data_dir);
        }

        self.graph = GraphBackend::Disk(Box::new(disk_graph));
        Ok(())
    }

    /// Sync column store references from DirGraph to DiskGraph.
    /// Called after enable_columnar(), add_nodes(), and load.
    pub fn sync_disk_column_stores(&mut self) {
        if let GraphBackend::Disk(ref mut dg) = self.graph {
            let mut stores = HashMap::new();
            for (type_name, store) in &self.column_stores {
                let key = InternedKey::from_str(type_name);
                stores.insert(key, Arc::clone(store));
            }
            dg.set_column_stores(stores);
        }
    }

    /// Mirror DiskGraph's column_stores back into DirGraph's
    /// `self.column_stores`. Called after mutations that flushed
    /// through `DiskGraph::flush_node_mut_cache` so the sidecar writer
    /// in `save_disk` and any other DirGraph-side reader sees the
    /// post-flush state rather than the pre-flush (stale) Arcs.
    pub fn sync_column_stores_from_disk(&mut self) {
        if let GraphBackend::Disk(ref dg) = self.graph {
            let pairs: Vec<(
                String,
                Arc<crate::graph::storage::column_store::ColumnStore>,
            )> = dg
                .column_stores_iter()
                .map(|(k, v)| (self.interner.resolve(*k).to_string(), Arc::clone(v)))
                .collect();
            for (name, arc) in pairs {
                self.column_stores.insert(name, arc);
            }
        }
    }

    /// Build CSR from pending edges if in disk mode. No-op otherwise.
    /// Called after add_connections, before queries, and before save.
    pub fn ensure_disk_edges_built(&mut self) {
        if let GraphBackend::Disk(ref mut dg) = self.graph {
            dg.build_csr_from_pending();
            // Don't compact here — overflow-merge is O(E), so calling it
            // after every add_connections batch would make multi-batch
            // builds quadratic. Queries still see overflow edges via the
            // merged DiskEdges iterator. Aggregate caches (conn_type_index
            // / peer_count_histogram) are refreshed at save time by
            // `save_disk` when overflow is present.
        }
    }

    /// Compact a disk-mode graph: merge overflow edges back into CSR arrays.
    /// Returns the number of overflow edges that were merged.
    /// No-op if there are no overflow edges.
    pub fn compact_disk(&mut self) -> Result<usize, String> {
        match &mut self.graph {
            GraphBackend::Disk(ref mut dg) => Ok(dg.compact()),
            _ => Err("compact requires disk mode".to_string()),
        }
    }

    /// Save a disk-mode graph to a directory. The directory IS the graph.
    /// Persists CSR files, node data, edge properties, column stores, and metadata.
    pub fn save_disk(&mut self, path: &str) -> Result<(), String> {
        // Build CSR from pending edges if not yet built.
        self.ensure_disk_edges_built();
        // Merge overflow edges back so conn_type_index and
        // peer_count_histogram reflect every live edge. Skipped during
        // builds; done here as a one-shot so users only pay the cost at
        // save time, not per add_connections batch.
        //
        // Gate: the phase-6 seal path in `save_to_dir` consumes
        // `overflow_out` / `overflow_in` directly. Running `compact()`
        // first moves those edges into the CSR (clearing overflow),
        // which causes seal to write an empty segment and lose the
        // new edges on reload. Only compact when we're taking the
        // compact-rewrite path (no prior save, or no tail above the
        // sealed watermark).
        if let GraphBackend::Disk(ref mut dg) = self.graph {
            let will_seal =
                !dg.segment_manifest.is_empty() && dg.sealed_nodes_bound < dg.node_count() as u32;
            if !will_seal && dg.has_overflow() {
                dg.compact();
            }
            // Auto-build the cross-type global title index so that
            // `MATCH (n {title: 'X'})` and `g.search(text)` are O(log N)
            // out of the box on every saved disk graph. Runs after
            // CSR / overflow consolidation so it sees the final node
            // set. Tied to `save_disk` rather than `build_csr_*` so
            // node-only graphs (no edges) still get the index built.
            let _ = dg.build_global_property_index("title");
            // Likewise index `nid` — the string id form for prefixed-id
            // datasets (Wikidata `"Q42"`). Since 0.11.0 `{nid: 'Q42'}` is a
            // plain string-property lookup (not the integer id-index), so the
            // index keeps it O(log N) instead of a 124M-row scan. No-op when
            // no type has a `nid` column.
            let _ = dg.build_global_property_index("nid");
        }

        let dir = std::path::Path::new(path);
        // save_to_dir needs &mut access so the edge-property store can
        // drop its base mmap before overwriting (PR2).
        let dg = match &mut self.graph {
            GraphBackend::Disk(dg) => dg,
            _ => return Err("save_disk requires disk mode".to_string()),
        };

        // Save DiskGraph files (CSR, nodes, edge properties, metadata).
        // `save_to_dir` runs `clear_arenas` internally, which drains
        // `node_mut_cache` via the clone-apply-replace flush, updating
        // each mutated type's Arc in `DiskGraph.column_stores`.
        dg.save_to_dir(dir, &self.interner)
            .map_err(|e| format!("DiskGraph save failed: {}", e))?;
        // Mirror the post-flush Arcs back into `self.column_stores` so
        // the per-type sidecar writer below sees the mutated stores
        // rather than the pre-flush (stale) Arcs. Pre-fix, mutations
        // landed in DiskGraph's Arcs but the sidecar writer read
        // DirGraph's Arcs — Cypher SET and DETACH DELETE property
        // corrections never reached disk.
        self.sync_column_stores_from_disk();

        // Save DirGraph metadata. 0.8.13 stripped `type_connectivity`;
        // 0.8.28 strips the two heavy HashMap fields
        // (`node_type_metadata`, `connection_type_metadata`) into
        // dedicated binary sidecars. The remaining metadata.json is
        // small (under a few hundred KB even on Wikidata-scale) and
        // parses in milliseconds.
        crate::graph::io::file::write_node_type_metadata_bin(dir, self)?;
        crate::graph::io::file::write_connection_type_metadata_bin(dir, self)?;
        // Secondary labels — disk's columnar layout has no slot for
        // NodeData.extra_labels, so we persist the inverted index as
        // a sidecar. Skipped when the graph has no secondaries
        // (single-label disk graphs pay zero extra bytes).
        crate::graph::io::file::write_secondary_labels_bin(dir, self)?;
        let mut meta = crate::graph::io::file::build_disk_metadata(self);
        crate::graph::io::file::strip_type_connectivity(&mut meta);
        crate::graph::io::file::strip_heavy_metadata(&mut meta);
        let meta_json = serde_json::to_string_pretty(&meta)
            .map_err(|e| format!("Metadata serialization failed: {}", e))?;
        std::fs::write(dir.join("metadata.json"), meta_json)
            .map_err(|e| format!("Failed to write metadata: {}", e))?;

        // Emit the packed binary `type_connectivity.bin.zst` at the
        // graph root; no-op when the cache is empty.
        crate::graph::io::file::write_type_connectivity_bin(dir, self)?;

        // 0.8.13: interner switches from JSON (hash → original) to
        // bincode `Vec<String>` of originals. The hash is re-derived
        // deterministically on load via `get_or_intern`. Loader falls
        // back to `interner.json` for graphs saved by 0.8.12 and
        // earlier.
        crate::graph::io::file::write_interner_bin(dir, self)?;

        // Save column stores (per type, sidecar format). Two modes:
        //
        // 1. **No `columns.bin`** — DirGraph → disk saves that never
        //    went through the N-Triples streaming builder. Write every
        //    column store as a sidecar.
        //
        // 2. **`columns.bin` exists** — N-Triples-built disk graphs.
        //    The single-file v3 layout covers every type that was
        //    present at ingest time, and reloading it via the mmap fast
        //    path (`file.rs:580`) is dramatically cheaper than walking
        //    per-type sidecars on a 88k-type wiki graph. BUT: types
        //    added post-build via `add_nodes` / `add_node` are not in
        //    `columns.bin` and were silently dropped on save before
        //    this fix. Emit sidecars *only* for those types — keeps the
        //    fast path for initial types, makes mutation persistence
        //    correct for new ones.
        //
        // 0.8.12 phase-1: PR1 phase 4 moved `columns.bin` under
        // `seg_000/`, so the presence check covers both the root (legacy
        // flat layout) and `seg_000/` (post-phase-4 segmented layout).
        // Mode-3 (new in 0.9.15): no preexisting `columns.bin` AND
        // we have in-memory `column_stores` (typical fresh save:
        // streaming carve, save_subset, mutation persist of an
        // in-memory build). Emit the unified mega-file format that
        // the loader's mmap fast path consumes — same layout the
        // ntriples builder produces — so the saved graph loads with
        // the same speed as a freshly-built one. Without this, a
        // saved DiskGraph fell into the per-type zstd sidecar path
        // and took ~70 s to load on a 17 M-node Wikidata carve vs.
        // ~150 ms for the full graph.
        let preexisting_columns_bin =
            dir.join("seg_000/columns.bin").exists() || dir.join("columns.bin").exists();
        if !preexisting_columns_bin && !self.column_stores.is_empty() {
            crate::graph::io::unified_columns::write_unified_columns(
                dir,
                &self.column_stores,
                &self.interner,
            )
            .map_err(|e| format!("unified columns write failed: {}", e))?;
        }

        let columns_meta_path = {
            let seg0_bin = dir.join("seg_000/columns_meta.bin.zst");
            let seg0_json = dir.join("seg_000/columns_meta.json");
            let root_bin = dir.join("columns_meta.bin.zst");
            let root_json = dir.join("columns_meta.json");
            if seg0_bin.exists() {
                Some(seg0_bin)
            } else if seg0_json.exists() {
                Some(seg0_json)
            } else if root_bin.exists() {
                Some(root_bin)
            } else if root_json.exists() {
                Some(root_json)
            } else {
                None
            }
        };
        let types_in_columns_bin: std::collections::HashSet<String> =
            if let Some(meta_path) = &columns_meta_path {
                use crate::graph::io::ntriples::ColumnTypeMeta;
                let metas: Vec<ColumnTypeMeta> =
                    if meta_path.extension().and_then(|s| s.to_str()) == Some("zst") {
                        let compressed = std::fs::read(meta_path)
                            .map_err(|e| format!("read {}: {}", meta_path.display(), e))?;
                        let bytes = zstd::decode_all(compressed.as_slice())
                            .map_err(|e| format!("decompress columns_meta: {}", e))?;
                        bincode::deserialize(&bytes)
                            .map_err(|e| format!("parse columns_meta.bin: {}", e))?
                    } else {
                        let json = std::fs::read_to_string(meta_path)
                            .map_err(|e| format!("read {}: {}", meta_path.display(), e))?;
                        serde_json::from_str(&json)
                            .map_err(|e| format!("parse columns_meta.json: {}", e))?
                    };
                metas.into_iter().map(|tm| tm.type_name).collect()
            } else {
                std::collections::HashSet::new()
            };

        let columns_dir = dir.join("columns");
        let mut sidecars_written = 0usize;
        for (type_name, store) in &self.column_stores {
            if types_in_columns_bin.contains(type_name) {
                continue; // covered by the fast mmap path on reload
            }
            if sidecars_written == 0 {
                std::fs::create_dir_all(&columns_dir)
                    .map_err(|e| format!("Failed to create columns dir: {}", e))?;
            }
            let type_dir = columns_dir.join(type_name);
            std::fs::create_dir_all(&type_dir)
                .map_err(|e| format!("Failed to create type dir: {}", e))?;
            let packed = store
                .write_packed(&self.interner)
                .map_err(|e| format!("Column pack failed: {}", e))?;
            // Prefix with a magic tag + the ColumnStore's row_count so
            // `load_column_sidecars` can pass the correct row count to
            // `ColumnStore::load_packed`. Pre-fix the loader derived
            // row_count from `type_indices[type].len()`, which counts
            // only *live* rows — after a DETACH DELETE that leaves
            // tombstoned rows in the store, the mismatch caused
            // `load_packed` to read column blobs at the wrong offsets
            // and produce garbage titles / null ages on reload.
            //
            // Format:
            //   magic: 8 bytes b"KGLCOLv1"
            //   row_count: u32 LE
            //   packed: existing `write_packed` output
            //
            // Old-format sidecars (no magic tag) stay loadable via a
            // fallback in the load path.
            let mut framed: Vec<u8> = Vec::with_capacity(12 + packed.len());
            framed.extend_from_slice(b"KGLCOLv1");
            framed.extend_from_slice(&store.row_count().to_le_bytes());
            framed.extend_from_slice(&packed);
            let compressed = zstd::encode_all(framed.as_slice(), 3)
                .map_err(|e| format!("Column compression failed: {}", e))?;
            std::fs::write(type_dir.join("columns.zst"), compressed)
                .map_err(|e| format!("Failed to write columns: {}", e))?;
            sidecars_written += 1;
        }

        // 0.8.13: type_indices uses a flat CSR binary keyed by interner
        // hashes. 0.8.28+: id_indices uses an mmap-resident raw `.bin`
        // layout — load reads via memory-mapped binary search, no eager
        // HashMap rebuild. Backward-compat loaders fall through to the
        // old bincode/zstd paths when the new file is absent.
        crate::graph::storage::disk::type_index::write_type_indices_bin(
            dir,
            &self.type_indices,
            &self.interner,
        )?;
        crate::graph::storage::disk::id_index::write_id_indices_bin(
            dir,
            &self.id_indices,
            &self.interner,
        )?;

        // Save embeddings if any (matches write_kgl behavior for in-memory saves)
        if !self.embeddings.is_empty() {
            let emb_bytes = bincode::serialize(&self.embeddings)
                .map_err(|e| format!("embeddings serialization failed: {}", e))?;
            let emb_compressed = zstd::encode_all(emb_bytes.as_slice(), 3)
                .map_err(|e| format!("embeddings compression failed: {}", e))?;
            std::fs::write(dir.join("embeddings.bin.zst"), emb_compressed)
                .map_err(|e| format!("Failed to write embeddings: {}", e))?;
        }

        // Save timeseries_store if any
        if !self.timeseries_store.is_empty() {
            let ts_bytes = bincode::serialize(&self.timeseries_store)
                .map_err(|e| format!("timeseries serialization failed: {}", e))?;
            let ts_compressed = zstd::encode_all(ts_bytes.as_slice(), 3)
                .map_err(|e| format!("timeseries compression failed: {}", e))?;
            std::fs::write(dir.join("timeseries.bin.zst"), ts_compressed)
                .map_err(|e| format!("Failed to write timeseries: {}", e))?;
        }

        Ok(())
    }

    /// Convert all node properties from Compact to Columnar storage.
    /// Properties are moved into per-type `ColumnStore` instances.
    /// This reduces memory usage by eliminating per-node `Value` enum overhead
    /// for homogeneous typed columns.
    ///
    /// Idempotent fast path: returns early when (a) every live node
    /// is already in `PropertyStorage::Columnar`, AND (b) every
    /// node's `Arc<ColumnStore>` is identical to the one in
    /// `graph.column_stores` for its type. Without this guard, a
    /// second `g.save()` after a successful first save runs the
    /// full `for node in graph` rebuild loop against already-
    /// Columnar properties — at wiki100m that's ~257 s
    /// (820 µs/node × 938 k nodes) — purely wasted work. Mapped
    /// graphs from `load_ntriples` are also already fully columnar
    /// (linked via `build_columns_direct`'s second-pass), so the
    /// same fast-path applies. 0.8.16.
    ///
    /// The Arc-pointer check is required because
    /// `PropertyStorage::insert` for a Columnar variant does
    /// `Arc::make_mut(store)` which forks the Arc when shared; the
    /// node's local Arc gets the update, `graph.column_stores`
    /// keeps the old. Without detecting that fork, an
    /// `add_nodes(conflict_handling="update")` followed by
    /// `g.save()` would silently drop the new properties. Walking
    /// every node short-circuits on the first non-columnar OR
    /// forked node, so the cost is O(N × cheap-match) at worst.
    pub fn enable_columnar(&mut self) {
        if !self.column_stores.is_empty() && self.is_columnar() {
            let interner = &self.interner;
            let column_stores = &self.column_stores;
            let any_drift = self
                .graph
                .node_indices()
                .filter_map(|idx| self.graph.node_weight(idx))
                .any(|n| match &n.properties {
                    PropertyStorage::Columnar { store, .. } => {
                        let type_str = n.node_type_str(interner);
                        match column_stores.get(type_str) {
                            Some(graph_store) => !Arc::ptr_eq(store, graph_store),
                            None => true,
                        }
                    }
                    _ => true,
                });
            if !any_drift {
                return;
            }
        }
        use crate::graph::storage::column_store::ColumnStore;

        // Ensure properties are compacted first
        if self.type_schemas.is_empty() {
            self.compact_properties();
        }

        // Build a ColumnStore per node type
        let mut stores: HashMap<String, ColumnStore> = HashMap::new();
        // Track row_id assignment per type
        let mut row_ids: HashMap<String, HashMap<NodeIndex, u32>> = HashMap::new();

        // Clean type_indices: remove entries for deleted/tombstoned nodes
        let graph_ref = &self.graph;
        self.type_indices
            .retain_all(|idx| graph_ref.node_weight(*idx).is_some());

        // First pass: create stores and push rows
        for (node_type, indices) in self.type_indices.iter() {
            let schema = match self.type_schemas.get(node_type) {
                Some(s) => Arc::clone(s),
                None => continue,
            };
            let meta = self
                .node_type_metadata
                .get(node_type)
                .cloned()
                .unwrap_or_default();

            let mut store = ColumnStore::new(schema, &meta, &self.interner);
            let mut type_row_ids = HashMap::with_capacity(indices.len());

            for idx in indices.iter() {
                if let Some(node) = self.graph.node_weight(idx) {
                    // Push id/title for every node. For Columnar nodes, read from
                    // the old column store. For Compact/Map nodes, use node.id/title.
                    // Always push id and title. For Columnar nodes, try old store first,
                    // fall back to node fields. For Compact/Map, use node fields directly.
                    let id_val = if let PropertyStorage::Columnar {
                        store: old_store,
                        row_id: old_row,
                    } = &node.properties
                    {
                        old_store
                            .get_id(*old_row)
                            .unwrap_or_else(|| node.id.clone())
                    } else {
                        node.id.clone()
                    };
                    let title_val = if let PropertyStorage::Columnar {
                        store: old_store,
                        row_id: old_row,
                    } = &node.properties
                    {
                        old_store
                            .get_title(*old_row)
                            .unwrap_or_else(|| node.title.clone())
                    } else {
                        node.title.clone()
                    };

                    store.push_id(&id_val);
                    store.push_title(&title_val);

                    // Collect properties from current storage
                    let pairs: Vec<(InternedKey, Value)> = match &node.properties {
                        PropertyStorage::Compact { schema, values } => schema
                            .slots
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &ik)| {
                                values.get(i).and_then(|v| {
                                    if matches!(v, Value::Null) {
                                        None
                                    } else {
                                        Some((ik, v.clone()))
                                    }
                                })
                            })
                            .collect(),
                        PropertyStorage::Map(map) => {
                            map.iter().map(|(&k, v)| (k, v.clone())).collect()
                        }
                        PropertyStorage::Columnar {
                            store: old_store,
                            row_id,
                        } => old_store.row_properties(*row_id),
                    };

                    let row_id = store.push_row(&pairs);
                    type_row_ids.insert(idx, row_id);
                }
            }

            stores.insert(node_type.to_string(), store);
            row_ids.insert(node_type.to_string(), type_row_ids);
        }

        // Spill to disk if over memory limit
        if let Some(limit) = self.memory_limit {
            let total: usize = stores.values().map(|s| s.heap_bytes()).sum();
            if total > limit {
                let spill_dir = self.spill_dir.clone().unwrap_or_else(|| {
                    std::env::temp_dir().join(format!(
                        "kglite_spill_{}_{:x}",
                        std::process::id(),
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_nanos()
                    ))
                });
                // Register spill dir for cleanup on drop
                if let Ok(mut dirs) = self.temp_dirs.lock() {
                    dirs.push(spill_dir.clone());
                }
                // Spill stores from largest to smallest until under limit
                let mut by_size: Vec<_> = stores
                    .iter()
                    .map(|(t, s)| (t.clone(), s.heap_bytes()))
                    .collect();
                by_size.sort_by_key(|s| std::cmp::Reverse(s.1));
                let mut remaining = total;
                for (type_name, bytes) in by_size {
                    if remaining <= limit {
                        break;
                    }
                    let type_dir = spill_dir.join(&type_name);
                    if let Some(store) = stores.get_mut(&type_name) {
                        if store
                            .materialize_to_files(&type_dir, &self.interner)
                            .is_ok()
                        {
                            remaining -= bytes;
                        }
                    }
                }
            }
        }

        // Wrap stores in Arc
        let arc_stores: HashMap<String, Arc<ColumnStore>> =
            stores.into_iter().map(|(t, s)| (t, Arc::new(s))).collect();

        // Second pass: replace PropertyStorage in each node
        for (node_type, type_row_ids) in &row_ids {
            if let Some(store) = arc_stores.get(node_type) {
                for (&idx, &row_id) in type_row_ids {
                    if let Some(node) = self.graph.node_weight_mut(idx) {
                        node.properties = PropertyStorage::Columnar {
                            store: Arc::clone(store),
                            row_id,
                        };
                        // id/title were pushed into the store's reserved
                        // __id__/__title__ columns in the first pass, so the
                        // inline copies are now redundant. Null them to the
                        // sentinel (NodeData::id()/title() read through to the
                        // store) — otherwise topology serialization writes
                        // every id/title twice (inline + column section),
                        // bloating the saved file by ~27 B/node. Mirrors the
                        // load path (io/file.rs) and the mapped batch path
                        // (mutation/batch.rs), which both null here.
                        node.id = Value::Null;
                        node.title = Value::Null;
                    }
                }
            }
        }

        self.column_stores = arc_stores;
    }

    /// Convert all Columnar properties back to Compact.
    /// Used before serialization to produce backward-compatible .kgl files.
    pub fn disable_columnar(&mut self) {
        let node_indices: Vec<NodeIndex> = self.graph.node_indices().collect();
        for node_idx in node_indices {
            let node = self.graph.node_weight_mut(node_idx).unwrap();
            if let PropertyStorage::Columnar { store, row_id } = &node.properties {
                let rid = *row_id;
                let pairs = store.row_properties(rid);
                // row_properties() excludes the reserved __id__/__title__
                // columns, so a null-sentinel node (set by enable_columnar /
                // load) would lose its id/title when we drop the Columnar
                // link. Pull them back from the store first.
                let restored_id = if matches!(node.id, Value::Null) {
                    store.get_id(rid)
                } else {
                    None
                };
                let restored_title = if matches!(node.title, Value::Null) {
                    store.get_title(rid)
                } else {
                    None
                };
                let type_str = node.node_type_str(&self.interner);
                if let Some(schema) = self.type_schemas.get(type_str) {
                    node.properties = PropertyStorage::from_compact(pairs, schema);
                } else {
                    // Fallback to Map
                    let map: HashMap<InternedKey, Value> = pairs.into_iter().collect();
                    node.properties = PropertyStorage::Map(map);
                }
                if let Some(v) = restored_id {
                    node.id = v;
                }
                if let Some(v) = restored_title {
                    node.title = v;
                }
            }
        }
        self.column_stores.clear();
    }

    /// Returns true if any nodes are using columnar storage.
    pub fn is_columnar(&self) -> bool {
        !self.column_stores.is_empty()
    }

    /// Ensure a ColumnStore exists for `node_type` with a schema covering all
    /// the keys in `type_schemas[node_type]`. If the schema has grown since the
    /// store was created, the store is rebuilt (existing data migrated).
    /// Call `ensure_type_schema_keys()` first to register new keys.
    pub fn ensure_column_store_for_push(
        &mut self,
        node_type: &str,
    ) -> &mut crate::graph::storage::column_store::ColumnStore {
        use crate::graph::storage::column_store::ColumnStore;

        let current_schema = self
            .type_schemas
            .get(node_type)
            .cloned()
            .unwrap_or_else(|| Arc::new(TypeSchema::new()));

        let need_create = if let Some(existing) = self.column_stores.get(node_type) {
            // Rebuild if the TypeSchema has more keys than the store's schema
            existing.schema().len() < current_schema.len()
        } else {
            true
        };

        if need_create {
            let meta = self
                .node_type_metadata
                .get(node_type)
                .cloned()
                .unwrap_or_default();

            if let Some(old_arc) = self.column_stores.remove(node_type) {
                // Migrate existing data to new store with extended schema
                let old_store = Arc::try_unwrap(old_arc).unwrap_or_else(|a| (*a).clone());
                let mut new_store = ColumnStore::new(current_schema, &meta, &self.interner);
                // Re-push all existing rows (including id/title columns)
                for row_id in 0..old_store.row_count() {
                    if let Some(id_val) = old_store.get_id(row_id) {
                        new_store.push_id(&id_val);
                    }
                    if let Some(title_val) = old_store.get_title(row_id) {
                        new_store.push_title(&title_val);
                    }
                    let props = old_store.row_properties(row_id);
                    new_store.push_row(&props);
                }
                self.column_stores
                    .insert(node_type.to_string(), Arc::new(new_store));
            } else {
                let store = ColumnStore::new(current_schema, &meta, &self.interner);
                self.column_stores
                    .insert(node_type.to_string(), Arc::new(store));
            }
        }

        Arc::make_mut(self.column_stores.get_mut(node_type).unwrap())
    }

    /// Ensure the TypeSchema for `node_type` contains all the given keys.
    /// Creates the schema if it doesn't exist, extends it if it does.
    pub fn ensure_type_schema_keys(&mut self, node_type: &str, keys: &[InternedKey]) {
        let schema = self
            .type_schemas
            .entry(node_type.to_string())
            .or_insert_with(|| Arc::new(TypeSchema::new()));
        let s = Arc::make_mut(schema);
        for &key in keys {
            s.add_key(key);
        }
    }

    /// Insert one node, routing storage by backend; returns the new index.
    ///
    /// - **Memory / mapped**: build a Compact `NodeData` on the shared
    ///   `TypeSchema` and `add_node` — the heap `StableDiGraph` keeps the
    ///   properties (today's path; unchanged behaviour).
    /// - **Disk**: the disk `add_node` stores only a slot and drops the
    ///   `NodeData` payload, so route id/title/properties through the per-type
    ///   `ColumnStore` first (the same mechanism `batch.rs::flush_chunk` uses
    ///   for bulk `add_nodes`): register schema keys, push id/title/row, then
    ///   `add_node` a `Columnar` slot and `update_row_id`.
    ///
    /// Used by Cypher `CREATE` (`executor::write::create_node`) so a single
    /// choke point gives uniform create semantics across modes. The caller
    /// owns id-index / type-index / property-index / metadata bookkeeping.
    ///
    /// **Disk note:** this does NOT push the mutated store to the disk
    /// read-side (`dg.column_stores`). The caller must call
    /// [`sync_disk_column_stores`](Self::sync_disk_column_stores) **once after
    /// the batch of inserts** — per-node syncing would share the store `Arc`
    /// and force every subsequent `ensure_column_store_for_push` to deep-clone
    /// it (O(store) per node).
    pub fn insert_node_routed(
        &mut self,
        id: Value,
        title: Value,
        node_type: &str,
        properties: HashMap<String, Value>,
    ) -> NodeIndex {
        if self.graph.is_disk() {
            // Register property types in node_type_metadata from the values we
            // have in hand. Do NOT read the node back for this: on disk the
            // columnar store isn't synced to the read-side (`dg.column_stores`)
            // until the end of the clause, so a read-back would see no properties
            // — and the metadata-driven column persistence would then drop them on
            // save (properties survive in-memory but vanish after save/reload).
            // Merge-upsert, so it composes with any later `ensure_type_metadata`.
            //
            // Memory/mapped skip this: the caller (`create_node`) runs
            // `ensure_type_metadata` against the read-back node, which produces
            // identical metadata. Doing both was redundant per-node work — the
            // bulk-CREATE regression introduced in 0.10.17.
            let prop_types: HashMap<String, String> = properties
                .iter()
                .map(|(k, v)| (k.clone(), v.type_name().to_string()))
                .collect();
            self.upsert_node_type_metadata(node_type, prop_types);

            // Pre-intern property keys (and node type) before borrowing stores.
            let interned_props: Vec<(InternedKey, Value)> = properties
                .iter()
                .map(|(k, v)| (self.interner.get_or_intern(k), v.clone()))
                .collect();
            // Sort for a deterministic schema slot order (see the memory branch
            // below) — `properties` HashMap iteration order is randomized.
            let mut keys: Vec<InternedKey> = interned_props.iter().map(|(k, _)| *k).collect();
            keys.sort_unstable_by_key(|k| k.as_u64());
            self.ensure_type_schema_keys(node_type, &keys);

            let row_id = {
                let store = self.ensure_column_store_for_push(node_type);
                store.push_id(&id);
                store.push_title(&title);
                store.push_row(&interned_props)
            };
            // The Arc borrow above has ended; clone the (now-extended) store
            // handle for the node's Columnar pointer.
            let store_arc = Arc::clone(
                self.column_stores
                    .get(node_type)
                    .expect("ensure_column_store_for_push just inserted it"),
            );
            let node_type_key = self.interner.get_or_intern(node_type);
            // id/title live in the ColumnStore (pushed above); the disk
            // `add_node` drops NodeData.id/title anyway and reads row_id out of
            // the Columnar variant. update_row_id re-stamps it for parity with
            // the bulk path (harmless if already correct).
            let node_data = NodeData {
                id,
                title,
                node_type: node_type_key,
                properties: PropertyStorage::Columnar {
                    store: store_arc,
                    row_id,
                },
            };
            let idx = GraphWrite::add_node(&mut self.graph, node_data);
            GraphWrite::update_row_id(&mut self.graph, idx, row_id);
            idx
        } else {
            // Memory / mapped: Compact NodeData on the shared TypeSchema.
            // Sort keys for a deterministic schema slot order — `properties`
            // HashMap iteration is randomized per process, which would make the
            // saved column order (and compressed .kgl bytes) non-reproducible.
            // InternedKey's FNV hash is stable across processes/versions.
            let mut interned_keys: Vec<InternedKey> = properties
                .keys()
                .map(|k| self.interner.get_or_intern(k))
                .collect();
            interned_keys.sort_unstable_by_key(|k| k.as_u64());
            self.ensure_type_schema_keys(node_type, &interned_keys);
            let schema = Arc::clone(
                self.type_schemas
                    .get(node_type)
                    .expect("ensure_type_schema_keys just inserted it"),
            );
            let node_data = NodeData::new_compact(
                id,
                title,
                node_type.to_string(),
                properties,
                &mut self.interner,
                &schema,
            );
            GraphWrite::add_node(&mut self.graph, node_data)
        }
    }

    /// Check heap usage of column stores and spill largest to disk if over limit.
    /// No-op if memory_limit is None or the backend is memory-mode.
    pub fn maybe_spill_columns(&mut self) {
        let limit = match self.memory_limit {
            Some(l) => l,
            None => return,
        };
        let total: usize = self.column_stores.values().map(|s| s.heap_bytes()).sum();
        if total <= limit {
            return;
        }

        let spill_dir = self.spill_dir.clone().unwrap_or_else(|| {
            std::env::temp_dir().join(format!(
                "kglite_spill_{}_{:x}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
            ))
        });
        // Cache spill_dir for future calls
        if self.spill_dir.is_none() {
            self.spill_dir = Some(spill_dir.clone());
        }
        // Register for cleanup on drop
        if let Ok(mut dirs) = self.temp_dirs.lock() {
            if !dirs.contains(&spill_dir) {
                dirs.push(spill_dir.clone());
            }
        }

        // Spill largest stores first until under limit
        let mut by_size: Vec<_> = self
            .column_stores
            .iter()
            .map(|(t, s)| (t.clone(), s.heap_bytes()))
            .collect();
        by_size.sort_by_key(|s| std::cmp::Reverse(s.1));
        let mut remaining = total;
        for (type_name, bytes) in by_size {
            if remaining <= limit {
                break;
            }
            let type_dir = spill_dir.join(&type_name);
            let store = Arc::make_mut(self.column_stores.get_mut(&type_name).unwrap());
            if store
                .materialize_to_files(&type_dir, &self.interner)
                .is_ok()
            {
                remaining -= bytes;
            }
        }
    }

    pub fn reindex(&mut self) {
        // 1. Rebuild type_indices from scratch
        self.rebuild_type_indices();

        // 2. Clear lazy caches — they'll rebuild on next access
        self.id_indices.clear();
        self.connection_types.clear();

        // 3. Rebuild existing property_indices (preserve which indexes exist)
        let property_keys: Vec<IndexKey> = self.property_indices.keys().cloned().collect();
        for (node_type, property) in property_keys {
            self.create_index(&node_type, &property);
        }

        // 4. Rebuild existing composite_indices (preserve which indexes exist)
        let composite_keys: Vec<CompositeIndexKey> =
            self.composite_indices.keys().cloned().collect();
        for (node_type, properties) in composite_keys {
            let prop_refs: Vec<&str> = properties.iter().map(|s| s.as_str()).collect();
            self.create_composite_index(&node_type, &prop_refs);
        }

        // 5. Rebuild existing range_indices (preserve which indexes exist)
        let range_keys: Vec<IndexKey> = self.range_indices.keys().cloned().collect();
        for (node_type, property) in range_keys {
            self.create_range_index(&node_type, &property);
        }
    }

    /// Compact the graph by removing tombstones left by deleted nodes/edges.
    ///
    /// With StableDiGraph, deletions leave holes (tombstones) in the internal
    /// storage. Over time, this wastes memory and degrades iteration performance.
    /// vacuum() rebuilds the graph with contiguous indices, then rebuilds all indexes.
    ///
    /// Returns a mapping from old NodeIndex → new NodeIndex so callers can
    /// update any external references (e.g., selections).
    ///
    /// No-op if there are no tombstones (node_count == node_bound).
    pub fn vacuum(&mut self) -> HashMap<NodeIndex, NodeIndex> {
        let old_node_count = self.graph.node_count();
        let old_node_bound = self.graph.node_bound();

        // No petgraph tombstones — but columnar stores may still have orphaned rows
        // (e.g., all nodes deleted → petgraph is empty but column data remains).
        if old_node_count == old_node_bound {
            let columnar_orphaned = self.column_stores.iter().any(|(t, s)| {
                let live = self.type_indices.get(t).map(|v| v.len()).unwrap_or(0);
                (s.row_count() as usize) > live
            });
            if columnar_orphaned {
                let saved_limit = self.memory_limit.take();
                self.disable_columnar();
                self.enable_columnar();
                self.memory_limit = saved_limit;
            }
            return HashMap::new();
        }

        // Build new graph with contiguous indices
        let mut new_graph = StableDiGraph::with_capacity(old_node_count, self.graph.edge_count());
        let mut old_to_new: HashMap<NodeIndex, NodeIndex> = HashMap::with_capacity(old_node_count);

        // Copy all live nodes, recording index mapping
        for old_idx in self.graph.node_indices() {
            let node_data = self.graph[old_idx].clone();
            let new_idx = new_graph.add_node(node_data);
            old_to_new.insert(old_idx, new_idx);
        }

        // Copy all live edges with remapped endpoints
        for old_edge_idx in self.graph.edge_indices() {
            if let Some((src, tgt)) = self.graph.edge_endpoints(old_edge_idx) {
                let edge_data = self.graph[old_edge_idx].clone();
                let new_src = old_to_new[&src];
                let new_tgt = old_to_new[&tgt];
                new_graph.add_edge(new_src, new_tgt, edge_data);
            }
        }

        // Replace graph storage
        self.graph = GraphBackend::Memory(MemoryGraph(new_graph));

        // Remap embedding stores to use new node indices
        for store in self.embeddings.values_mut() {
            let mut new_node_to_slot = HashMap::with_capacity(store.node_to_slot.len());
            let mut new_slot_to_node = Vec::with_capacity(store.slot_to_node.len());
            let mut new_data = Vec::with_capacity(store.data.len());

            for (&old_node_raw, &slot) in &store.node_to_slot {
                let old_idx = NodeIndex::new(old_node_raw);
                if let Some(&new_idx) = old_to_new.get(&old_idx) {
                    let new_slot = new_slot_to_node.len();
                    new_node_to_slot.insert(new_idx.index(), new_slot);
                    new_slot_to_node.push(new_idx.index());
                    let start = slot * store.dimension;
                    let end = start + store.dimension;
                    new_data.extend_from_slice(&store.data[start..end]);
                }
                // Deleted nodes (not in old_to_new) are dropped
            }

            store.node_to_slot = new_node_to_slot;
            store.slot_to_node = new_slot_to_node;
            store.data = new_data;
            // Slots were remapped wholesale; resync the cached-norm column.
            store.rebuild_norms();
        }

        // Rebuild all indexes from the compacted graph
        self.reindex();

        // Rebuild columnar stores if active — old stores have orphaned rows
        // from deleted nodes. The disable/enable cycle reads only live nodes,
        // producing fresh ColumnStores with no dead rows.
        if !self.column_stores.is_empty() {
            let saved_limit = self.memory_limit.take();
            self.disable_columnar();
            self.enable_columnar();
            self.memory_limit = saved_limit;
        }

        old_to_new
    }

    /// Check if auto-vacuum should run and trigger it if so.
    ///
    /// Called after DELETE operations. Only vacuums if:
    /// - `auto_vacuum_threshold` is Some(threshold)
    /// - Tombstones exceed 100 (avoid overhead on tiny graphs)
    /// - `fragmentation_ratio` exceeds the threshold
    ///
    /// Returns true if vacuum was triggered.
    pub fn check_auto_vacuum(&mut self) -> bool {
        let threshold = match self.auto_vacuum_threshold {
            Some(t) => t,
            None => return false,
        };

        let node_count = self.graph.node_count();
        let node_bound = self.graph.node_bound();
        let tombstones = node_bound - node_count;

        if tombstones <= 100 {
            return false;
        }

        let ratio = tombstones as f64 / node_bound as f64;
        if ratio > threshold {
            self.vacuum();
            true
        } else {
            false
        }
    }

    /// Return diagnostic information about graph storage health.
    ///
    /// Useful for deciding when to call vacuum():
    /// - `tombstones` > 0 means deleted nodes left holes
    /// - `fragmentation_ratio` approaching 1.0 means most storage is wasted
    /// - A ratio above 0.3 is a good threshold for calling vacuum()
    pub fn graph_info(&self) -> GraphInfo {
        let node_count = self.graph.node_count();
        let node_bound = self.graph.node_bound();
        let edge_count = self.graph.edge_count();
        let node_tombstones = node_bound - node_count;

        GraphInfo {
            node_count,
            node_capacity: node_bound,
            node_tombstones,
            edge_count,
            fragmentation_ratio: if node_bound == 0 {
                0.0
            } else {
                node_tombstones as f64 / node_bound as f64
            },
            type_count: self.type_indices.len(),
            property_index_count: self.property_indices.len(),
            composite_index_count: self.composite_indices.len(),
            columnar_total_rows: self
                .column_stores
                .values()
                .map(|s| s.row_count() as usize)
                .sum(),
            columnar_live_rows: self
                .column_stores
                .keys()
                .map(|t| self.type_indices.get(t).map(|v| v.len()).unwrap_or(0))
                .sum(),
        }
    }
}

/// Statistics about a property index
#[derive(Debug, Clone)]
pub struct IndexStats {
    pub unique_values: usize,
    pub total_entries: usize,
    pub avg_entries_per_value: f64,
}

/// Diagnostic information about graph storage health.
#[derive(Debug, Clone)]
pub struct GraphInfo {
    /// Number of live nodes in the graph
    pub node_count: usize,
    /// Upper bound of node indices (includes tombstones from deletions)
    pub node_capacity: usize,
    /// Number of tombstone slots (node_capacity - node_count)
    pub node_tombstones: usize,
    /// Number of live edges in the graph
    pub edge_count: usize,
    /// Ratio of wasted storage (0.0 = clean, approaching 1.0 = heavily fragmented)
    pub fragmentation_ratio: f64,
    /// Number of distinct node types
    pub type_count: usize,
    /// Number of single-property indexes
    pub property_index_count: usize,
    /// Number of composite indexes
    pub composite_index_count: usize,
    /// Total rows across all columnar stores (including orphaned from deletions)
    pub columnar_total_rows: usize,
    /// Rows backed by live nodes (columnar_total_rows - columnar_live_rows = orphaned)
    pub columnar_live_rows: usize,
}

/// Get a `&mut DirGraph` from an `Arc<DirGraph>` and bump the version
/// counter. Wraps [`Arc::make_mut`] (which clones the inner `DirGraph`
/// if other strong refs exist) plus the canonical post-mutation
/// `version += 1` increment that downstream OCC commit-checks rely on.
///
/// Lifted from the wheel crate in 0.10.1 so bindings + embedders that
/// hold an `Arc<DirGraph>` and want to mutate it have a single,
/// consistent entry point.
///
/// **Warning:** If other `Arc<DirGraph>` references exist (e.g. a
/// snapshot held by an open transaction, or a clone held by a still-
/// alive `ResultView`), this deep-clones the entire graph — every
/// node, edge, and index. Mutation in a read-heavy workload is fine,
/// but a lingering reference can cause an unexpected memory spike on
/// the first write.
pub fn make_dir_graph_mut(arc: &mut std::sync::Arc<DirGraph>) -> &mut DirGraph {
    let graph = std::sync::Arc::make_mut(arc);
    graph.version += 1;
    graph
}

#[cfg(test)]
mod multi_label_tests {
    use super::*;
    use crate::datatypes::Value;
    use crate::graph::schema::NodeData;
    use crate::graph::storage::GraphWrite;

    fn add_node(graph: &mut DirGraph, id: &str, node_type: &str) -> NodeIndex {
        let nd = NodeData::new(
            Value::String(id.to_string()),
            Value::String(id.to_string()),
            node_type.to_string(),
            HashMap::new(),
            &mut graph.interner,
        );
        let idx = GraphWrite::add_node(&mut graph.graph, nd);
        graph
            .type_indices
            .entry_or_default(node_type.to_string())
            .push(idx);
        idx
    }

    #[test]
    fn add_node_label_idempotent_and_no_op_on_primary() {
        let mut g = DirGraph::new();
        let idx = add_node(&mut g, "n1", "Person");
        let reviewer = g.interner.get_or_intern("Reviewer");
        let person = g.interner.get_or_intern("Person");

        assert!(g.add_node_label(idx, reviewer));
        assert!(g.has_secondary_labels);
        assert_eq!(g.secondary_label_index[&reviewer], vec![idx]);

        // Idempotent — second add is a no-op.
        assert!(!g.add_node_label(idx, reviewer));
        assert_eq!(g.secondary_label_index[&reviewer], vec![idx]);

        // Primary type is a no-op too.
        assert!(!g.add_node_label(idx, person));

        let labels = g.node_labels(idx);
        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0], person);
        assert_eq!(labels[1], reviewer);
    }

    #[test]
    fn remove_node_label_errors_on_primary() {
        let mut g = DirGraph::new();
        let idx = add_node(&mut g, "n1", "Person");
        let person = g.interner.get_or_intern("Person");

        let err = g.remove_node_label(idx, person).unwrap_err();
        assert!(err.contains("primary label"));
    }

    #[test]
    fn remove_node_label_clears_index_when_last_node_drops_it() {
        let mut g = DirGraph::new();
        let a = add_node(&mut g, "a", "Person");
        let b = add_node(&mut g, "b", "Person");
        let reviewer = g.interner.get_or_intern("Reviewer");

        g.add_node_label(a, reviewer);
        g.add_node_label(b, reviewer);
        assert_eq!(g.secondary_label_index[&reviewer].len(), 2);

        assert!(g.remove_node_label(a, reviewer).unwrap());
        assert_eq!(g.secondary_label_index[&reviewer], vec![b]);
        assert!(g.has_secondary_labels);

        assert!(g.remove_node_label(b, reviewer).unwrap());
        assert!(!g.secondary_label_index.contains_key(&reviewer));
        // No labels left anywhere, fast-skip resets.
        assert!(!g.has_secondary_labels);
    }

    #[test]
    fn rebuild_does_not_clobber_secondary_index() {
        // After 0.10.5's perf fix, NodeData no longer carries
        // extra_labels — `secondary_label_index` is the canonical
        // store. `rebuild_type_indices` rebuilds only type_indices
        // and leaves the secondary index intact (it's repopulated by
        // the load path via the disk sidecar / .kgl section).
        let mut g = DirGraph::new();
        let idx = add_node(&mut g, "n1", "Person");
        let reviewer = g.interner.get_or_intern("Reviewer");
        g.add_node_label(idx, reviewer);

        let before = g.secondary_label_index.clone();
        let before_flag = g.has_secondary_labels;

        g.rebuild_type_indices();

        // Secondary index is untouched.
        assert_eq!(g.secondary_label_index, before);
        assert_eq!(g.has_secondary_labels, before_flag);
        // Primary type_indices is rebuilt correctly.
        assert_eq!(
            g.type_indices.get("Person").map(|s| s.iter().collect()),
            Some(vec![idx])
        );
    }

    #[test]
    fn dir_graph_node_labels_returns_primary_plus_extras() {
        // The canonical path for "all labels of node X" is
        // `DirGraph::node_labels` (which scans `secondary_label_index`).
        // Backend trait `node_labels_of` returns only the primary
        // type and is no longer the authoritative source.
        let mut g = DirGraph::new();
        let idx = add_node(&mut g, "n1", "Person");
        let reviewer = g.interner.get_or_intern("Reviewer");
        let person = g.interner.get_or_intern("Person");
        g.add_node_label(idx, reviewer);

        let labels = g.node_labels(idx);
        assert_eq!(labels, vec![person, reviewer]);
    }

    #[test]
    fn nodes_with_label_single_label_fast_path() {
        // With no secondary labels anywhere, nodes_with_label must
        // return exactly type_indices[label] — the byte-identical
        // result every primary-only call site produced pre-multi-label.
        let mut g = DirGraph::new();
        let a = add_node(&mut g, "a", "Person");
        let b = add_node(&mut g, "b", "Person");
        add_node(&mut g, "w", "Widget");

        assert!(!g.has_secondary_labels);
        assert_eq!(g.nodes_with_label("Person"), vec![a, b]);
        assert_eq!(g.nodes_with_label("Widget").len(), 1);
        assert!(g.nodes_with_label("Absent").is_empty());
    }

    #[test]
    fn nodes_with_label_unions_primary_and_secondary() {
        let mut g = DirGraph::new();
        let a = add_node(&mut g, "a", "Person"); // primary Person, + VIP
        let b = add_node(&mut g, "b", "Person"); // primary Person only
        let w = add_node(&mut g, "w", "Widget"); // primary Widget, + VIP
        let vip = g.interner.get_or_intern("VIP");
        g.add_node_label(a, vip);
        g.add_node_label(w, vip);

        // Primary lookups still include their primary-typed nodes.
        let persons = g.nodes_with_label("Person");
        assert_eq!(persons, vec![a, b]);

        // :VIP is a secondary-only label — union pulls from both buckets.
        let mut vips = g.nodes_with_label("VIP");
        vips.sort();
        let mut expected = vec![a, w];
        expected.sort();
        assert_eq!(vips, expected);
    }

    #[test]
    fn node_has_label_primary_secondary_and_absent() {
        let mut g = DirGraph::new();
        let a = add_node(&mut g, "a", "Person");
        let person = g.interner.get_or_intern("Person");
        let vip = g.interner.get_or_intern("VIP");
        let ghost = g.interner.get_or_intern("Ghost");
        g.add_node_label(a, vip);

        assert!(g.node_has_label(a, person)); // primary
        assert!(g.node_has_label(a, vip)); // secondary
        assert!(!g.node_has_label(a, ghost)); // absent
    }

    #[test]
    fn detach_delete_evicts_secondary_label_index() {
        use std::collections::HashSet;
        let mut g = DirGraph::new();
        let a = add_node(&mut g, "a", "Person");
        let b = add_node(&mut g, "b", "Person");
        let vip = g.interner.get_or_intern("VIP");
        g.add_node_label(a, vip);
        g.add_node_label(b, vip);
        assert_eq!(g.secondary_label_index[&vip].len(), 2);

        let to_del: HashSet<NodeIndex> = [a].into_iter().collect();
        crate::graph::mutation::maintain::detach_delete_nodes(&mut g, &to_del);

        // `a` evicted from the secondary index; `b` survives. Without the
        // eviction the StableDiGraph would keep `a` live in the bucket and
        // `nodes_with_label` / counts would over-report.
        assert_eq!(g.secondary_label_index.get(&vip).map(|v| v.len()), Some(1));
        assert!(g.has_secondary_labels);
        assert_eq!(g.nodes_with_label("VIP"), vec![b]);
    }
}
