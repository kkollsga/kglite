//! DirGraph — transactional container for the in-memory graph.
//!
//! Owns the `StableDiGraph` + all type/property/composite/range indexes,
//! OCC `version`, `schema_locked`, spatial / temporal / timeseries configs,
//! embedding stores, connection-type metadata, and schema definitions.

use self::index_layer::LayeredIndex;
use self::range_index_layer::LayeredRangeIndex;
use crate::datatypes::values::Value;
use crate::graph::constraints::{NamedConstraint, UniqueConstraintKey};
use crate::graph::property_types::DeclaredType;
use crate::graph::schema::{
    CompositeIndexKey, CompositeValue, ConnectionTypeInfo, ConnectivityTriple, EdgeData,
    EmbeddingStore, GraphBackend, IndexKey, InternedKey, NodeData, PropertyStorage, SaveMetadata,
    SchemaDefinition, SpatialConfig, StringInterner, TemporalConfig, TypeIdIndex, TypeSchema,
};
use crate::graph::storage::column_store::ColumnStore;
use crate::graph::storage::disk::id_index::IdIndexStore;
use crate::graph::storage::disk::type_index::TypeIndexStore;

// Counts full `enable_columnar` rebuilds, so the save fast path can be pinned
// by measurement rather than by argument. Thread-local because `cargo test`
// runs in parallel and a global would count other tests' rebuilds.
#[cfg(test)]
thread_local! {
    pub(crate) static COLUMNAR_REBUILDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
#[inline]
fn note_columnar_rebuild() {
    COLUMNAR_REBUILDS.with(|c| c.set(c.get() + 1));
}
use crate::graph::storage::{GraphRead, GraphWrite};
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::stable_graph::StableDiGraph;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

/// Source of process-unique graph ids. Starts at 1 (0 is never handed out, so
/// it can serve as a sentinel); monotonic and never reused so a dropped graph's
/// plan-cache entries can't be served to a later graph that reuses an address.
static NEXT_GRAPH_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Mint a fresh, process-unique graph id (also the serde default for the
/// skipped `graph_id` field, so a loaded graph gets a new identity).
pub(crate) fn next_graph_id() -> u64 {
    NEXT_GRAPH_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

// Lazily-computed caches + derived stats (edge-type counts, type
// connectivity, per-(type,property) NDV) live in a child module so the
// file stays under the god-file ceiling; child = retains private access.
pub(crate) mod caches;
mod columnar_rebuild;
pub mod constraints;
mod disk_persistence;
mod independent_copy;
pub mod index_layer;
pub(crate) mod indexes;
mod labels;
pub mod node_remap;
mod node_write;
pub mod ontology_apply;
pub mod range_index_layer;
pub(crate) mod rel_constraints;
pub(crate) mod rollback;
pub(crate) mod schema_cow;
mod schema_ops;

pub use node_remap::NodeRemap;

/// Version-keyed cache of per-`(type, property)` distinct-value counts (NDV)
/// — see the [`DirGraph::property_ndv_cache`] field.
type PropertyNdvCache = Arc<RwLock<(u64, HashMap<(String, String), usize>)>>;

/// Core graph storage: a directed graph (petgraph `StableDiGraph`) with fast
/// type-based indexing and optional property/composite/range/spatial indexes.
#[derive(Clone, Serialize, Deserialize)]
pub struct DirGraph {
    pub graph: GraphBackend,
    /// Skipped during serialization — rebuilt from graph on load via `rebuild_type_indices()`.
    /// On disk graphs the base layer is mmap-backed via `type_indices.bin`;
    /// mutations land in an in-memory overlay.
    #[serde(skip)]
    pub type_indices: TypeIndexStore,
    #[serde(default)]
    pub schema_definition: Option<SchemaDefinition>,
    /// Single-property indexes for fast lookups: (node_type, property) -> value -> [node_indices]
    /// Skipped during serialization — rebuilt from `property_index_keys` on load.
    ///
    /// Each index's `value -> members` map is a [`LayeredIndex`]: a stack of
    /// shared, immutable levels, so a fork shares the buckets instead of
    /// copying one `Value` key and one `Vec` per distinct value (48.0 ms at 1M
    /// before layering). Reads and edits keep the `HashMap` shape.
    #[serde(skip)]
    pub property_indices: HashMap<IndexKey, LayeredIndex<Value>>,
    /// Composite indexes for multi-field queries: (node_type, [properties]) -> composite_value -> [node_indices]
    /// Skipped during serialization — rebuilt from `composite_index_keys` on load.
    ///
    /// [`LayeredIndex`] for the same reason as `property_indices`, and more
    /// urgently: a `CompositeValue` key is a `Vec<Value>`, so the fork it
    /// replaces allocated a `Vec` plus a `String` per component per distinct
    /// tuple — 88.9 ms at 1M, the largest single term anywhere in the fork.
    #[serde(skip)]
    pub composite_indices: HashMap<CompositeIndexKey, LayeredIndex<CompositeValue>>,
    /// Persisted list of property index keys so indexes can be rebuilt on load
    #[serde(default)]
    pub property_index_keys: Vec<IndexKey>,
    /// Persisted list of composite index keys so indexes can be rebuilt on load
    #[serde(default)]
    pub composite_index_keys: Vec<CompositeIndexKey>,
    /// B-Tree range indexes for ordered lookups: (node_type, property) -> value -> [node_indices]
    /// Skipped during serialization — rebuilt from `range_index_keys` on load.
    ///
    /// A [`LayeredRangeIndex`]: `index_layer`'s level stack over **ordered**
    /// levels, so a fork shares the B-tree instead of copying it while
    /// `lookup_range` still walks values in order. Held-view first write at
    /// 100k measured 0.889 ms against an equality index's 0.048 ms before this
    /// (P4) — the gap was the copy.
    #[serde(skip)]
    pub range_indices: HashMap<IndexKey, LayeredRangeIndex<Value>>,
    /// Persisted list of range index keys so indexes can be rebuilt on load
    #[serde(default)]
    pub range_index_keys: Vec<IndexKey>,
    /// Declared UNIQUE constraints, as the enforcement structure itself:
    /// (node_type, [properties]) -> tuple value -> the one node occupying it.
    /// A single-occupant map (rather than the `Vec<NodeIndex>` the other index
    /// kinds carry) *is* the constraint: an occupied slot is the violation.
    /// Skipped during serialization — rebuilt from `unique_constraint_keys` on
    /// load, which re-verifies the constraint as a side effect.
    #[serde(skip)]
    pub(crate) unique_indices: HashMap<UniqueConstraintKey, HashMap<CompositeValue, NodeIndex>>,
    /// Persisted list of declared unique constraints so they survive
    /// save/load. Additive serde field — older `.kgl` files load with an empty
    /// list, i.e. no constraints, which is the pre-existing behaviour.
    #[serde(default)]
    pub(crate) unique_constraint_keys: Vec<UniqueConstraintKey>,
    /// Which declared unique constraints came from DDL
    /// (`CREATE CONSTRAINT ... IS UNIQUE` / `... IS NODE KEY`) rather than from
    /// a schema, with the property tuple normalised (sorted, deduped) so a
    /// re-spelling of the same constraint hits the same entry.
    ///
    /// A DDL declaration and a schema `primary_key`/`unique` on the same
    /// `(node_type, properties)` share **one** entry in `unique_indices` — the
    /// index *is* the constraint, so there is no second copy to withdraw
    /// independently. Without this record, withdrawing the schema half took the
    /// whole index with it: `define_schema` naming the type without its key
    /// deleted a `CREATE CONSTRAINT` the caller never touched, and duplicates
    /// were admitted. `set_schema` consults it (`withdraw_schema_unique`) and
    /// retains an index a DDL declaration still backs.
    ///
    /// Written only by the DDL entry points, so a schema install cannot forge
    /// provenance: [`Self::declare_ddl_unique_constraint`] records,
    /// [`Self::drop_unique_constraint`] and
    /// [`Self::drop_unique_constraints_for_type`] forget.
    ///
    /// **Persisted through `FileMetadata`, not through this derive** — see
    /// [`Self::ddl_not_null_constraints`], which carries the same provenance
    /// role for the presence half and the same additive-field posture: a file
    /// written before this field loads with an empty set, leaving its
    /// DDL-declared uniqueness indistinguishable from a schema-declared one,
    /// exactly as it was.
    #[serde(default)]
    pub(crate) ddl_unique_constraints: std::collections::BTreeSet<UniqueConstraintKey>,
    /// User-supplied constraint names → the declaration each one names, so
    /// `DROP CONSTRAINT <name>` resolves. KGLite's enforcement structures are
    /// keyed by `(node_type, properties)`, so a Neo4j-style constraint name has
    /// nowhere else to live; this registry is a lookup aid and never the source
    /// of truth (see `NamedConstraint` and `prune_constraint_names`). Additive
    /// serde field — older `.kgl` files load with an empty map, which only means
    /// their constraints must be dropped by descriptor.
    #[serde(default)]
    pub(crate) constraint_names: HashMap<String, NamedConstraint>,
    /// `(node_type, property)` presence constraints declared through DDL
    /// (`CREATE CONSTRAINT ... IS NOT NULL`) rather than through a schema.
    ///
    /// The enforced list itself lives in `SchemaDefinition::required_fields`, so
    /// without this provenance record an unrelated `define_schema` would replace
    /// the schema and silently un-enforce a DDL declaration — the asymmetry that
    /// does not exist for uniqueness, whose index lives outside the schema.
    /// `set_schema` replays this set over the newly installed schema. A
    /// `BTreeSet` so the persisted bytes are deterministic.
    ///
    /// **Persisted through `FileMetadata`, not through this derive.** A `.kgl`
    /// load builds a *fresh* `DirGraph` and repopulates it from the metadata
    /// struct (`io::file`: `from_graph` / `apply_to_with`), so a `serde`
    /// attribute here carries nothing across a save on its own — the field was
    /// silently lost on every reload until `FileMetadata` gained its
    /// counterpart, which let the first `define_schema()` after a load
    /// un-enforce a DDL-declared constraint. Additive there: a file written
    /// before that field loads with an empty set, which only means its DDL
    /// presence constraints are indistinguishable from schema-declared ones,
    /// exactly as they were.
    #[serde(default)]
    pub(crate) ddl_not_null_constraints: std::collections::BTreeSet<(String, String)>,
    /// Declared property-type constraints (`CREATE CONSTRAINT … IS :: T`), as
    /// `node_type -> property -> type`.
    ///
    /// Unlike the presence half, this needs no `reapply_*` counterpart: it is
    /// the enforcement structure *and* the provenance record, and it lives
    /// outside `SchemaDefinition`, so installing a schema cannot silently
    /// un-enforce it — the same arrangement uniqueness gets from
    /// `unique_indices`.
    ///
    /// Nested `BTreeMap` rather than the flat `BTreeSet` of tuples its sibling
    /// uses, for three reasons: a declaration carries a *value* (the type),
    /// lookup is per `(node_type, property)` on the write path rather than a
    /// membership test, and string keys keep the shape serialisable as JSON as
    /// well as postcard (a tuple key is not a JSON object key). Ordered, so the
    /// bytes are deterministic once it is persisted.
    ///
    /// **Persisted through `FileMetadata`, not through this derive** — see
    /// [`Self::ddl_not_null_constraints`]. This map *is* the enforcement
    /// structure — nothing else remembers the declared type — so losing it
    /// across a save would silently stop enforcing every declaration.
    #[serde(default)]
    pub(crate) ddl_property_type_constraints:
        std::collections::BTreeMap<String, std::collections::BTreeMap<String, DeclaredType>>,
    /// Declared *relationship* presence constraints, as
    /// `(connection_type, property)`.
    ///
    /// Its own store rather than an entry in the node sibling: a connection
    /// type and a node type share a namespace in neither direction, and the
    /// two are read by different write paths. Unlike the node half this map
    /// *is* the constraint — a connection type has no `required_fields` list
    /// inside `SchemaDefinition` to mirror a declaration into, so nothing
    /// needs a `reapply_*` pass and no schema install can withdraw one.
    ///
    /// **Persisted through `FileMetadata`**, like every constraint store here
    /// (`io::file`: `from_graph` / `apply_to_with`); this derive carries
    /// nothing across a save on its own.
    #[serde(default)]
    pub(crate) rel_ddl_not_null_constraints: std::collections::BTreeSet<(String, String)>,
    /// Declared relationship property-type constraints, as
    /// `connection_type -> property -> type`. The relationship counterpart of
    /// [`Self::ddl_property_type_constraints`], with the same shape and the
    /// same persistence model.
    #[serde(default)]
    pub(crate) rel_ddl_property_type_constraints:
        std::collections::BTreeMap<String, std::collections::BTreeMap<String, DeclaredType>>,
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
    /// Replaces SchemaNode graph nodes — persisted via versioned binary Serde.
    ///
    /// `Arc`-shared: the rollback shell copies the pointer, not the
    /// catalogue — see `schema_cow` for the copy-on-write contract and the
    /// measurement that earned it. Read through the field (it derefs); write
    /// only through [`DirGraph::node_type_metadata_mut`].
    #[serde(default)]
    pub node_type_metadata: Arc<HashMap<String, HashMap<String, String>>>,
    /// Connection type metadata: connection_type → ConnectionTypeInfo
    /// Replaces SchemaNode graph nodes for connections — persisted via versioned binary Serde.
    ///
    /// `Arc`-shared for the rollback shell — see `schema_cow`; write only
    /// through [`DirGraph::connection_type_metadata_mut`].
    #[serde(default)]
    pub connection_type_metadata: Arc<HashMap<String, ConnectionTypeInfo>>,
    /// Version and library info stamped at save time.
    /// Old files without this field deserialize to SaveMetadata::default() (format_version=0).
    #[serde(default)]
    pub save_metadata: SaveMetadata,
    /// Original ID field name per node type (e.g. "Person" → "npdid").
    /// Stored when the user-supplied unique_id_field differs from "id".
    /// Used for alias resolution: querying by original column name maps to the `id` field.
    ///
    /// `Arc`-shared for the rollback shell — see `schema_cow`; write only
    /// through [`DirGraph::id_field_aliases_mut`].
    #[serde(default)]
    pub id_field_aliases: Arc<FxHashMap<String, String>>,
    /// Original title field name per node type (e.g. "Person" → "prospect_name").
    /// Stored when the user-supplied node_title_field differs from "title".
    /// Used for alias resolution: querying by original column name maps to the `title` field.
    ///
    /// `Arc`-shared for the rollback shell — see `schema_cow`; write only
    /// through [`DirGraph::title_field_aliases_mut`].
    #[serde(default)]
    pub title_field_aliases: Arc<FxHashMap<String, String>>,
    /// Parent type for supporting node types: child_type → parent_type.
    /// If a type has an entry here, it is a "supporting" type that belongs to the parent.
    /// Types without an entry are "core" types (shown in describe() inventory).
    ///
    /// `Arc`-shared for the rollback shell — see `schema_cow`; write only
    /// through [`DirGraph::parent_types_mut`].
    #[serde(default)]
    pub parent_types: Arc<HashMap<String, String>>,
    /// The declared semantic layer (`graph/ontology.rs`) — an `is_a` forest
    /// plus relationship semantics. Deliberately independent of
    /// `parent_types` (semantic "kind of" vs presentation ownership; see the
    /// ontology module doc). `Arc`-shared like `parent_types` so forks and
    /// the rollback shell share it O(1). Persisted via `FileMetadata`.
    #[serde(
        default,
        skip_serializing_if = "crate::graph::ontology::arc_store_is_empty"
    )]
    pub ontology: Arc<crate::graph::ontology::OntologyStore>,
    /// Column-order + dtype fidelity for table-valued properties
    /// (`set_table_property`): `(node_type, property)` → ordered column
    /// names plus their declared dtypes. The stored value stays a plain
    /// `list<map>` (PropMap keys are sorted, so order lives here); Cypher
    /// never consults this — only the DataFrame reconstruction does.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub table_property_meta:
        std::collections::BTreeMap<String, crate::graph::tables::TablePropertyMeta>,
    /// Declared structured property shapes (`tables.rs`), keyed by
    /// `table_meta_key(node_type, property)`. Declared through
    /// `define_schema` `types` values that use the shape grammar; enforced
    /// at the write gates (never at WAL replay).
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub property_shapes: std::collections::BTreeMap<String, crate::graph::tables::PropertyShape>,
    /// Materialized-label bookkeeping (`ontology_apply.rs`): label →
    /// Closed/Open. Persisted via `FileMetadata` beside the store.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub managed_labels:
        std::collections::BTreeMap<String, crate::graph::ontology::ManagedLabelState>,
    /// Derived per-type ancestor cache (`ontology_apply.rs`); rebuilt from
    /// the store on define/clear/load, never persisted.
    #[serde(skip)]
    pub(crate) ontology_closures: HashMap<InternedKey, Vec<InternedKey>>,
    /// WAL replay sets this while it rebuilds state: the write-funnel
    /// closure stamp must not run there — the log's whole-set label ops are
    /// authoritative, and re-deriving would un-apply a logged
    /// dematerialize. Transient, never persisted.
    #[serde(skip)]
    pub(crate) suppress_ontology_stamp: bool,
    /// Free-text instructions/briefing rendered verbatim at the top of
    /// `describe()` so an agent opening the graph cold sees how to use it.
    /// Keyed by channel; the empty string `""` is the default channel (the
    /// only one the v1 API writes). Storing a map keeps per-audience channels
    /// a trivial v2 without changing the format. Additive — absent in old files.
    #[serde(default)]
    pub graph_instructions: HashMap<String, String>,
    /// **User**-schema version — the caller's own data-model revision, bumped by
    /// their migrations. Distinct from the engine's format stamps
    /// (`save_metadata.format_version`, the `.kgl` magic), which the engine owns
    /// and this never touches. `0` = unversioned, which is also what a `.kgl`
    /// written before this field existed loads as (additive, absent in old
    /// files). Docs: `docs/python/guides/schema-migrations.md`.
    #[serde(default)]
    pub user_schema_version: u32,
    /// Highest WAL log-sequence number already folded into this graph's last
    /// checkpoint — the replay gate for a durable session.
    ///
    /// A durable `save()` stamps the LSN of the newest logged frame here before
    /// serializing, so the `.kgl` records *how far* the snapshot has consumed
    /// the log. On the next durable open, replay skips every frame at or below
    /// it, which is what makes recovery robust to a **stale WAL prefix** — a log
    /// whose surviving frames predate the checkpoint. Folding such a prefix over
    /// a newer snapshot would roll properties back to an earlier commit and
    /// destroy already-durable data.
    ///
    /// `0` = "no checkpoint has consumed the log", which is also what a `.kgl`
    /// written before this field existed loads as: replay everything, the
    /// pre-gate behaviour. The counter is monotonic for the life of the log and
    /// is **not** reset by a checkpoint — a per-checkpoint reset would make
    /// every stamped value 0 and the gate vacuous, and would let a stale frame
    /// carry the same LSN as a fresh one.
    #[serde(default)]
    pub checkpoint_lsn: u64,
    /// What the last save recorded about the change log that was running then
    /// — see [`CdcHandoff`](crate::graph::cdc::CdcHandoff).
    ///
    /// Restored from the file's metadata on load, like `checkpoint_lsn` above,
    /// and read by exactly one thing: `db.cdc.query`'s wrong-epoch refusal,
    /// which upgrades to naming where the old epoch ended when the cursor it
    /// was handed belongs to the stamped one. Capture itself stays **off** on
    /// load; this is a diagnostic about a log that is gone, never a means of
    /// resuming it.
    ///
    /// `#[serde(default)]` is vestigial here as for its neighbours — the load
    /// path repopulates from `FileMetadata`, so it only keeps in-memory clones
    /// and older payloads working.
    #[serde(default)]
    pub cdc_handoff: Option<crate::graph::cdc::CdcHandoff>,
    /// Auto-vacuum threshold: if Some(t), vacuum() is triggered automatically
    /// after DELETE operations when the worst of the three garbage populations
    /// (node slots, dead columnar rows, edge slots) exceeds 100 and the worst
    /// of their ratios exceeds t — [`Self::check_auto_vacuum`] owns the
    /// arithmetic. Default: Some(0.3). Set to None to disable.
    #[serde(default = "default_auto_vacuum_threshold")]
    pub auto_vacuum_threshold: Option<f64>,
    /// How many times [`Self::check_auto_vacuum`] has fired a vacuum on this
    /// in-memory graph.
    ///
    /// Counts *fired* vacuums, not reclaimed slots — a fired vacuum that found
    /// nothing still increments, because the trigger arithmetic is what this
    /// measures. Lifetime of the graph object, not of the file: the `.kgl`
    /// writer does not carry it, so a reopened graph starts at 0.
    #[serde(default)]
    pub auto_vacuums_run: u64,
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
    ///
    /// **Fork-private** — see [`caches::ForkPrivateCache`] for the aliasing
    /// bug that earned it (a snapshot reporting the writer's edge counts) and
    /// for why `wkt_cache` and `property_ndv_cache` deliberately stay shared.
    #[serde(skip)]
    pub edge_type_counts_cache: caches::ForkPrivateCache<Arc<HashMap<String, usize>>>,
    /// Cached type connectivity: (source_type, connection_type, target_type) → count.
    /// Computed by `rebuild_caches()`, persisted in metadata, restored on load.
    /// Invalidated on edge mutations alongside edge_type_counts_cache.
    /// Fork-private, for the same reason as `edge_type_counts_cache`.
    #[serde(skip)]
    pub type_connectivity_cache: caches::ForkPrivateCache<Vec<ConnectivityTriple>>,
    /// Lazy per-`(type, property)` distinct-value count (NDV), used by the
    /// planner to estimate non-indexed equality selectivity
    /// (`type_count / ndv`) instead of a flat heuristic. The tuple's `u64` is
    /// the graph `version` the map was built at: a mismatch means a mutation
    /// happened, so the map is dropped and recomputed — auto-invalidation
    /// without per-mutation-site bookkeeping. Plan-time read path only.
    #[serde(skip)]
    pub property_ndv_cache: PropertyNdvCache,
    /// Columnar embedding storage: (node_type, property_name) -> EmbeddingStore.
    /// Stored separately from NodeData.properties — invisible to normal node API.
    /// Persisted as a separate section in v2 .kgl files.
    #[serde(skip)]
    pub embeddings: HashMap<(String, String), EmbeddingStore>,
    /// Lexical (BM25) text indexes: (node_type, property) -> TextIndexStore.
    /// Opt-in, built explicitly via `build_text_index`, and heap-resident like
    /// the embedding stores beside them — see [`crate::graph::text_indexes`]
    /// for the slot convention and the invalidation rules, and
    /// [`crate::graph::index_freshness`] for how a store catches up with writes
    /// that landed after its build.
    ///
    /// Parked by `rollback::swap_data_scale` for the same reason `embeddings`
    /// is: it is corpus-sized, so a statement checkpoint must not clone it.
    #[serde(skip)]
    pub text_indexes: HashMap<(String, String), crate::graph::text_indexes::TextIndexStore>,
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
    /// Transient, **execution-scoped** write whitelist (role-scoped writes —
    /// integrity, not secrecy). When `Some(set)`, a Cypher **node** write is
    /// rejected unless the node's *stored* type is in `set`, and a
    /// **relationship** write is rejected unless at least one endpoint's
    /// stored type is; the full perimeter is documented on
    /// [`crate::graph::session::execute::ExecuteOptions::write_scope`] and
    /// enforced by the `enforce_*_write_scope` family in
    /// `languages::cypher::executor::write_scope`. Set by `execute_mut` for the
    /// duration of one mutation and cleared immediately after; never a
    /// persistent graph property and never serialized. `None` = unrestricted
    /// (the default; zero cost).
    #[serde(skip, default)]
    pub(crate) active_write_scope: Option<std::collections::HashSet<String>>,
    /// Caller-supplied freshness provenance for the current mutation: the git
    /// commit SHA the writer is working against (`active_git_sha`) and an actor
    /// id (`active_modified_by`). Stamped alongside `updated_at` on writes to
    /// `auto_timestamp` types. Set by `execute_mut` for one mutation and cleared
    /// immediately after (same lifecycle as `active_write_scope`); never
    /// serialized. `None` = not supplied (the default; zero cost).
    #[serde(skip, default)]
    pub(crate) active_git_sha: Option<String>,
    #[serde(skip, default)]
    pub(crate) active_modified_by: Option<String>,
    /// Transient, **execution-scoped** carrier for the structured constraint
    /// violation behind the write error currently unwinding.
    ///
    /// The Cypher mutation tree (`executor/write.rs`, `executor/schema_ddl.rs`)
    /// and the bulk-loader gate (`mutation/maintain.rs`) both report failures
    /// over a `Result<_, String>` channel, so a `ConstraintViolation` — which
    /// already has an `impl From<ConstraintViolation> for KgError` — would
    /// otherwise be flattened to prose and surface as a generic
    /// `CypherExecutionError` / `ArgumentError`. Rather than retype ~535 call
    /// sites across the executor, the violation is parked here by
    /// [`Self::record_constraint_violation`] at the moment it is stringified
    /// and drained by the adapter that builds the typed error.
    ///
    /// Stored **with the exact message it produced**; the drain
    /// ([`Self::take_constraint_violation_for`]) only accepts a byte-identical
    /// string, so a desync fails *safe* rather than *wrong*.
    ///
    /// Same lifecycle as `active_write_scope`: installed for one execution,
    /// cleared unconditionally, never serialized, never copied (see
    /// `independent_copy`).
    #[serde(skip, default)]
    pub(crate) pending_constraint_violation:
        Option<Box<(String, crate::graph::constraints::ConstraintViolation)>>,
    /// Monotonically increasing version counter — incremented on every mutation.
    /// Used for optimistic concurrency control in transactions.
    #[serde(skip, default)]
    pub version: u64,
    /// Process-unique graph identity, assigned at construction. Never persisted
    /// — a loaded `.kgl` is a fresh runtime instance and gets a new id. Used
    /// with `version` as the Cypher plan-cache key so a cached plan can never
    /// leak across graphs.
    ///
    /// **Preserved by `Clone`, re-minted whenever a clone becomes an
    /// independently mutable lineage** — `independent_copy` and
    /// `fork_transaction`. Plain `Clone` keeps the id because it backs
    /// snapshots and CoW views, which are the *same* state until something
    /// diverges; a transaction working copy is precisely that divergence, and
    /// leaving it on the parent's id made two sibling forks indistinguishable
    /// to the plan cache (see `session::transaction::fork_transaction`).
    /// `version` alone does not distinguish them: siblings bump in lockstep.
    #[serde(skip, default = "next_graph_id")]
    pub graph_id: u64,
    /// Change-data-capture log, when a caller has enabled it — the bounded
    /// ring of published [`CdcEvent`](crate::graph::cdc::CdcEvent)s plus its
    /// `(epoch, seq)` addressing.
    ///
    /// **Never serialized**, like `graph_id` and for the same reason: it is a
    /// runtime identity a consumer holds cursors into, and a `.kgl` that
    /// carried one would hand those cursors to a *copy* of the data in another
    /// process, where the same `seq` means something else. A load therefore
    /// starts with capture off. The `DirGraph` serde derive is vestigial for
    /// the data itself (a `.kgl` persists the backend, columns and file
    /// metadata), so this field costs nothing at save either way.
    ///
    /// **Shared by `Clone`** — a copy-on-write view, a transaction fork and
    /// the graph they came from publish into one log, which is what makes a
    /// commit taken over a held reader land exactly once. `independent_copy`
    /// re-mints a fresh log with a new epoch, as it re-mints `graph_id`.
    #[serde(skip)]
    pub(crate) cdc: Option<crate::graph::cdc::CdcHandle>,
    /// High-water mark for engine-minted node ids — see
    /// [`DirGraph::next_auto_node_id`]. Never serialized: it is re-derived
    /// from `node_bound()` on first use, so a loaded `.kgl` starts above the
    /// ids it holds without persisting (or format-breaking for) a counter.
    #[serde(skip, default)]
    pub(crate) next_auto_id: u32,
    /// Property key interner: maps InternedKey(u64) → original string.
    /// Populated during ingestion (add_nodes, CREATE, SET) and deserialization.
    /// Skipped during serde — rebuilt on load by the InternedKey Deserialize impl.
    #[serde(skip)]
    pub interner: StringInterner,
    /// Shared property schemas per node type: type_name → Arc<TypeSchema>.
    /// Populated during ingestion (add_nodes, CREATE) and compaction (load).
    ///
    /// `Arc`-shared for the rollback shell — see `schema_cow`; write only
    /// through [`DirGraph::type_schemas_mut`].
    #[serde(skip)]
    pub type_schemas: Arc<HashMap<String, Arc<TypeSchema>>>,
    /// Fast-skip flag: true if any node has secondary labels.
    /// Read paths short-circuit the secondary_label_index scan entirely
    /// when this is false, so single-label graphs pay no perf tax.
    /// `#[serde(skip)]` — maintained by the same writers as
    /// [`Self::secondary_label_index`] (the choke-point label API, the load
    /// path, WAL replay); `rebuild_type_indices` leaves both alone.
    #[serde(skip)]
    pub has_secondary_labels: bool,
    /// O(1) secondary-label index: label_key → [NodeIndex]. **The
    /// canonical store** — `NodeData` carries no labels of its own, so this
    /// map is the only record that a node has any secondary label.
    ///
    /// Written exclusively by the choke-point label mutation API
    /// (`DirGraph::add_node_label` / `remove_node_label`), which is also
    /// where rollback and WAL capture hook in for the same reason: the map
    /// sits above the storage backend, so no `GraphWrite` call can carry a
    /// label change.
    ///
    /// `#[serde(skip)]` — it cannot be derived from node payloads, so it is
    /// persisted out-of-band and restored by the load path: the `.kgl`
    /// `secondary_labels` section for in-memory graphs, the
    /// `secondary_labels.bin.zst` sidecar for disk ones, and
    /// `MutationOp::SetNodeLabels` frames for post-checkpoint WAL replay.
    /// `rebuild_type_indices` deliberately leaves it alone.
    #[serde(skip)]
    pub secondary_label_index: HashMap<InternedKey, Vec<NodeIndex>>,
}

pub(crate) fn default_auto_vacuum_threshold() -> Option<f64> {
    Some(0.3)
}

impl Drop for DirGraph {
    fn drop(&mut self) {
        // Only the last Arc holder removes the shared temp dirs.
        if let Ok(dirs) = self.temp_dirs.lock() {
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

/// The keys of a hash map, sorted — the shape every persisted index-key
/// snapshot takes (see [`DirGraph::populate_index_keys`]).
fn sorted_keys<K: Ord + Clone, V>(map: &HashMap<K, V>) -> Vec<K> {
    let mut keys: Vec<K> = map.keys().cloned().collect();
    keys.sort_unstable();
    keys
}

impl DirGraph {
    /// Current monotonic version counter, bumped by every mutation path. The
    /// OCC token for [`crate::graph::session`] and downstream consumers (the
    /// Python `Transaction` class, the `kglite-bolt-server` per-tx commit
    /// path). Exposed publicly via `kglite::api::DirGraph::version`.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Process-unique graph identity (see the `graph_id` field). Pairs with
    /// [`Self::version`] to form the Cypher plan-cache key.
    pub fn graph_id(&self) -> u64 {
        self.graph_id
    }

    /// This graph's change-data-capture log, or `None` when capture is off.
    /// Lifecycle and publishing live in [`crate::graph::cdc`].
    pub fn cdc_log(&self) -> Option<&crate::graph::cdc::CdcHandle> {
        self.cdc.as_ref()
    }

    /// Whether change data capture is enabled on this graph.
    pub fn cdc_enabled(&self) -> bool {
        self.cdc.is_some()
    }

    /// Mint the next **engine-assigned** node id, for a `CREATE` that supplied
    /// no `id` property.
    ///
    /// ## Why this is not `node_bound()`
    ///
    /// It used to be. `StableDiGraph::node_bound` is an *index-space* bound,
    /// not an allocator: it shrinks when the highest-indexed nodes are removed,
    /// and it stalls while `add_node` refills previously-freed slots. So both
    /// of these minted a live id a second time — silently, since the per-type
    /// id index is a map and simply drops the loser:
    ///
    /// - `CREATE`×5, `DELETE` two, `CREATE`×3 → two nodes with id 5.
    /// - `CREATE`, `DELETE`, `save()`, `load()`, `CREATE`×3 → *three* nodes
    ///   sharing one id.
    ///
    /// That is engine-side identity corruption, and it is distinct from the
    /// documented "uniqueness is opt-in" rule (CYPHER.md), which governs ids
    /// the **caller** supplies: a caller who writes `{id: 1}` twice has asked
    /// for two nodes and gets a `duplicate_id`-auditable graph, whereas a
    /// caller who supplies no id has asked the engine for an identity and must
    /// get a distinct one. It is also how the defect reaches durability: WAL
    /// replay folds ops by `(node_type, id)`, so a duplicate id makes recovery
    /// silently *merge* two nodes the live graph kept apart.
    ///
    /// The mark is a session-scoped high-water line rather than a persisted
    /// counter: `max`-ing against `node_bound()` on every call re-seeds it
    /// above a freshly loaded graph's own ids for free, so no `.kgl` field —
    /// and no postcard format break — is needed. In the common no-delete case
    /// the two advance in lockstep and the ids handed out are byte-identical
    /// to the old ones (0, 1, 2, …).
    pub(crate) fn next_auto_node_id(&mut self) -> Value {
        let candidate = self.next_auto_id.max(self.graph.node_bound() as u32);
        self.next_auto_id = candidate.saturating_add(1);
        Value::UniqueId(candidate)
    }

    /// Keep the auto-id high-water mark above an id the *caller* supplied, so
    /// a later engine-minted id cannot land on it. Cheap enough (one compare)
    /// to call from every write path that accepts caller ids; a non-integer id
    /// shares no value space with `UniqueId` and is ignored.
    pub(crate) fn observe_explicit_id(&mut self, id: &Value) {
        let seen = match id {
            Value::UniqueId(u) => *u,
            Value::Int64(i) if *i >= 0 && *i <= u32::MAX as i64 => *i as u32,
            _ => return,
        };
        self.next_auto_id = self.next_auto_id.max(seen.saturating_add(1));
    }

    /// Set the version directly. Used by [`crate::graph::session::Session::commit`]
    /// to bump the working DirGraph's version on commit-swap. Not
    /// for general use — mutation paths bump version through their
    /// own mechanisms.
    pub fn set_version(&mut self, v: u64) {
        self.version = v;
    }

    /// Advance the version by one — the canonical "this graph just mutated"
    /// signal. Every mutation path routes through this (Cypher writes via
    /// `execute_mut`, bulk ingest, and the `make_dir_graph_mut` handle) so
    /// version-keyed caches (the Cypher plan cache) and OCC observe every
    /// change.
    pub fn bump_version(&mut self) {
        self.version = self.version.wrapping_add(1);
    }

    /// Stringify `violation` for the `Result<_, String>` error channel while
    /// parking the structured value in
    /// [`Self::pending_constraint_violation`], so the adapter that ultimately
    /// builds the typed error can recover the constraint kind, node type,
    /// properties, and descriptor instead of only the prose.
    ///
    /// Returns the message to hand to `Err(..)`; call it at every site that
    /// would otherwise write `.map_err(|v| v.to_string())`.
    pub(crate) fn record_constraint_violation(
        &mut self,
        violation: crate::graph::constraints::ConstraintViolation,
    ) -> String {
        let message = violation.to_string();
        self.pending_constraint_violation = Some(Box::new((message.clone(), violation)));
        message
    }

    /// Clear any parked violation. Called before an execution begins so a
    /// violation left by an earlier run on the same working copy can never be
    /// attributed to a later, unrelated error.
    pub(crate) fn clear_pending_constraint_violation(&mut self) {
        self.pending_constraint_violation = None;
    }

    /// Take the parked violation **only if** it is the one behind `message`.
    ///
    /// The identity check is what makes the side channel safe: the `String` is
    /// still the control-flow channel, so if any intermediate frame wrapped or
    /// replaced the message, the parked violation no longer describes the error
    /// being reported and is dropped rather than mis-attributed. This is an
    /// equality test against a string this graph itself produced — not a
    /// pattern match on error prose.
    pub(crate) fn take_constraint_violation_for(
        &mut self,
        message: &str,
    ) -> Option<crate::graph::constraints::ConstraintViolation> {
        let parked = self.pending_constraint_violation.take()?;
        let (recorded, violation) = *parked;
        (recorded == message).then_some(violation)
    }

    /// Typed [`KgError`] for a write that failed with `message`, when that
    /// failure was a declared-constraint violation.
    ///
    /// Returns `None` when the error was something else, so each caller keeps
    /// its own fallback: the Cypher path degrades to
    /// [`KgError::CypherExecution`], the bulk-loader path to
    /// [`KgError::Argument`]. Every binding that surfaces a write error over
    /// the engine's `Result<_, String>` channel needs exactly this step, so it
    /// lives here rather than being re-derived per binding.
    pub fn take_constraint_error(&mut self, message: &str) -> Option<crate::error::KgError> {
        self.take_constraint_violation_for(message)
            .map(crate::error::KgError::from)
    }

    pub fn new() -> Self {
        DirGraph {
            graph_id: next_graph_id(),
            cdc: None,
            next_auto_id: 0,
            graph: GraphBackend::new(),
            type_indices: TypeIndexStore::new(),
            schema_definition: None,
            property_indices: HashMap::new(),
            composite_indices: HashMap::new(),
            property_index_keys: Vec::new(),
            composite_index_keys: Vec::new(),
            range_indices: HashMap::new(),
            range_index_keys: Vec::new(),
            unique_indices: HashMap::new(),
            ddl_unique_constraints: std::collections::BTreeSet::new(),
            unique_constraint_keys: Vec::new(),
            constraint_names: HashMap::new(),
            ddl_not_null_constraints: std::collections::BTreeSet::new(),
            ddl_property_type_constraints: std::collections::BTreeMap::new(),
            rel_ddl_not_null_constraints: std::collections::BTreeSet::new(),
            rel_ddl_property_type_constraints: std::collections::BTreeMap::new(),
            id_indices: IdIndexStore::new(),
            connection_types: std::collections::HashSet::new(),
            node_type_metadata: Arc::new(HashMap::new()),
            connection_type_metadata: Arc::new(HashMap::new()),
            save_metadata: SaveMetadata::current(),
            id_field_aliases: Arc::default(),
            title_field_aliases: Arc::default(),
            parent_types: Arc::new(HashMap::new()),
            ontology: Arc::default(),
            table_property_meta: std::collections::BTreeMap::new(),
            property_shapes: std::collections::BTreeMap::new(),
            managed_labels: std::collections::BTreeMap::new(),
            ontology_closures: HashMap::new(),
            suppress_ontology_stamp: false,
            graph_instructions: HashMap::new(),
            user_schema_version: 0,
            checkpoint_lsn: 0,
            cdc_handoff: None,
            auto_vacuum_threshold: default_auto_vacuum_threshold(),
            auto_vacuums_run: 0,
            spatial_configs: HashMap::new(),
            wkt_cache: Arc::new(RwLock::new(HashMap::new())),
            edge_type_counts_cache: Default::default(),
            type_connectivity_cache: Default::default(),
            property_ndv_cache: Arc::new(RwLock::new((0, HashMap::new()))),
            embeddings: HashMap::new(),
            text_indexes: HashMap::new(),
            timeseries_configs: HashMap::new(),
            timeseries_store: HashMap::new(),
            temporal_node_configs: HashMap::new(),
            temporal_edge_configs: HashMap::new(),
            memory_limit: None,
            spill_dir: None,
            temp_dirs: Arc::new(std::sync::Mutex::new(Vec::new())),
            read_only: false,
            schema_locked: false,
            active_write_scope: None,
            active_git_sha: None,
            active_modified_by: None,
            pending_constraint_violation: None,
            version: 0,
            interner: StringInterner::new(),
            type_schemas: Arc::new(HashMap::new()),
            has_secondary_labels: false,
            secondary_label_index: HashMap::new(),
        }
    }

    /// Create a DirGraph from a pre-existing graph (used by v3 loader).
    /// All metadata fields start empty and are populated by the caller.
    pub fn from_graph(graph: GraphBackend) -> Self {
        DirGraph {
            graph_id: next_graph_id(),
            cdc: None,
            next_auto_id: 0,
            graph,
            type_indices: TypeIndexStore::new(),
            schema_definition: None,
            property_indices: HashMap::new(),
            composite_indices: HashMap::new(),
            property_index_keys: Vec::new(),
            composite_index_keys: Vec::new(),
            range_indices: HashMap::new(),
            range_index_keys: Vec::new(),
            unique_indices: HashMap::new(),
            ddl_unique_constraints: std::collections::BTreeSet::new(),
            unique_constraint_keys: Vec::new(),
            constraint_names: HashMap::new(),
            ddl_not_null_constraints: std::collections::BTreeSet::new(),
            ddl_property_type_constraints: std::collections::BTreeMap::new(),
            rel_ddl_not_null_constraints: std::collections::BTreeSet::new(),
            rel_ddl_property_type_constraints: std::collections::BTreeMap::new(),
            id_indices: IdIndexStore::new(),
            connection_types: std::collections::HashSet::new(),
            node_type_metadata: Arc::new(HashMap::new()),
            connection_type_metadata: Arc::new(HashMap::new()),
            save_metadata: SaveMetadata::default(),
            id_field_aliases: Arc::default(),
            title_field_aliases: Arc::default(),
            parent_types: Arc::new(HashMap::new()),
            ontology: Arc::default(),
            table_property_meta: std::collections::BTreeMap::new(),
            property_shapes: std::collections::BTreeMap::new(),
            managed_labels: std::collections::BTreeMap::new(),
            ontology_closures: HashMap::new(),
            suppress_ontology_stamp: false,
            graph_instructions: HashMap::new(),
            user_schema_version: 0,
            checkpoint_lsn: 0,
            cdc_handoff: None,
            auto_vacuum_threshold: default_auto_vacuum_threshold(),
            auto_vacuums_run: 0,
            spatial_configs: HashMap::new(),
            wkt_cache: Arc::new(RwLock::new(HashMap::new())),
            edge_type_counts_cache: Default::default(),
            type_connectivity_cache: Default::default(),
            property_ndv_cache: Arc::new(RwLock::new((0, HashMap::new()))),
            embeddings: HashMap::new(),
            text_indexes: HashMap::new(),
            timeseries_configs: HashMap::new(),
            timeseries_store: HashMap::new(),
            temporal_node_configs: HashMap::new(),
            temporal_edge_configs: HashMap::new(),
            memory_limit: None,
            spill_dir: None,
            temp_dirs: Arc::new(std::sync::Mutex::new(Vec::new())),
            read_only: false,
            schema_locked: false,
            active_write_scope: None,
            active_git_sha: None,
            active_modified_by: None,
            pending_constraint_violation: None,
            version: 0,
            interner: StringInterner::new(),
            type_schemas: Arc::new(HashMap::new()),
            has_secondary_labels: false,
            secondary_label_index: HashMap::new(),
        }
    }

    pub fn get_spatial_config(&self, node_type: &str) -> Option<&SpatialConfig> {
        self.spatial_configs.get(node_type)
    }

    pub fn get_node_timeseries(
        &self,
        node_index: usize,
    ) -> Option<&crate::graph::features::timeseries::NodeTimeseries> {
        self.timeseries_store.get(&node_index)
    }

    /// Look up an embedding store by `(&str, &str)`: a linear scan of the
    /// embeddings map (typically 1-3 entries), so no owned `String` key has to
    /// be allocated for a `HashMap` probe.
    #[inline]
    pub fn embedding_store(&self, node_type: &str, prop_name: &str) -> Option<&EmbeddingStore> {
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

    /// Fold the `created` nodes a bulk append just added to `node_type` into
    /// that type's id index, instead of rebuilding it from every member.
    ///
    /// **The append is O(created), not O(type).** `add_nodes` used to drop the
    /// type's entry and rebuild it — one node read and `Value` clone per node
    /// of the type — on every call, so appending ten rows to a 200k-row type
    /// cost 10 ms of index work for 10 rows of data (perf scan 2026-08-14 #2).
    ///
    /// The created nodes are the **tail** of the type's bucket: the batch
    /// appends one member per creation, in creation order, and nothing else
    /// writes that bucket while it runs. Reading the tail rather than
    /// threading the pairs back through the batch engine keeps the delta out
    /// of memory entirely — a 1M-row ingest would otherwise carry a 1M-entry
    /// side vector purely to re-say what the bucket already records.
    ///
    /// Returns `false` when the fold cannot be trusted — the type is not
    /// indexed (folding into nothing would leave a *partial* entry that
    /// `build_id_index` short-circuits on and trusts as complete: the hazard
    /// documented at `mutation/batch.rs`), or the tail does not read back as
    /// `created` live nodes of this type. The caller must then invalidate and
    /// rebuild. `add_nodes` satisfies the first precondition by building the
    /// index before its row loop — it needs it there anyway — so that arm is a
    /// guard on the contract, not the expected path.
    pub(crate) fn fold_appended_ids_into_index(&mut self, node_type: &str, created: usize) -> bool {
        if !self.id_indices.contains_key(node_type) {
            return false;
        }
        if created == 0 {
            return true;
        }
        let Some(tail) = self.appended_tail(node_type, created) else {
            return false;
        };

        let type_key = InternedKey::from_str(node_type);
        let mut pairs: Vec<(Value, NodeIndex)> = Vec::with_capacity(created);
        {
            // Arena guard: `node_view` materializes on the disk backend.
            let _guard = self.graph.begin_query();
            for node_idx in tail {
                match self.graph.node_view(node_idx) {
                    Some(node) if node.node_type() == type_key => {
                        pairs.push((node.id().into_owned(), node_idx))
                    }
                    _ => return false,
                }
            }
        }

        let mut collisions = 0usize;
        let entry = self.id_indices.entry_or_default(node_type.to_string());
        for (id, node_idx) in pairs {
            // A created row whose id the index already resolves is a duplicate
            // the rebuild would have collapsed — and warned about. Kept because
            // the warning is the only signal a user gets that
            // `MATCH (n {id: …})` will now return one of two nodes.
            if entry.get(&id).is_some() {
                collisions += 1;
            }
            entry.insert(id, node_idx);
        }
        warn_on_duplicate_ids(node_type, created, created - collisions);
        true
    }

    /// The `created` members a bulk append just pushed onto `node_type`'s
    /// bucket, in creation order — or `None` when the bucket is too short to
    /// have carried them, which means the caller's count and this bucket
    /// disagree and nothing derived from it can be trusted.
    ///
    /// The batch appends exactly one member per creation and nothing else
    /// writes the bucket while it runs, so the tail *is* the delta.
    pub(crate) fn appended_tail(&self, node_type: &str, created: usize) -> Option<Vec<NodeIndex>> {
        let members = self.type_indices.get(node_type)?;
        if members.len() < created {
            return None;
        }
        let tail: Vec<NodeIndex> = (members.len() - created..members.len())
            .filter_map(|position| members.get(position))
            .collect();
        (tail.len() == created).then_some(tail)
    }

    /// `&self` counterpart of [`build_id_index`](Self::build_id_index):
    /// pre-warm the id index for a type through the `IdIndexStore`'s
    /// interior mutability. Same effect, but callable on a shared
    /// `Arc<DirGraph>` without `Arc::make_mut` (which would deep-copy the
    /// whole graph when another handle shares the Arc). No-op if the type
    /// is already indexed.
    pub fn ensure_id_index(&self, node_type: &str) {
        self.id_indices
            .ensure(node_type, || self.compute_id_index(node_type));
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
            None => return TypeIdIndex::General(FxHashMap::default()),
        };

        let mut all_unique_id = true;
        let mut entries: Vec<(Value, NodeIndex)> = Vec::with_capacity(node_indices.len());

        // Disk + column store: read ids directly from mmap'd columns.
        let used_columns = if let GraphBackend::Disk(ref dg) = self.graph {
            if let Some(store) =
                GraphRead::column_store(dg.as_ref(), InternedKey::from_str(node_type))
            {
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
            // Arena guard: on the disk backend node_weight materializes into
            // the query arena, which must run under a DiskQueryGuard (arena
            // protocol in disk/graph.rs, enforced by a debug assert); no-op
            // on memory/mapped backends.
            let _guard = self.graph.begin_query();
            for node_idx in node_indices.iter() {
                if let Some(node) = self.graph.node_view(node_idx) {
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
            let map: FxHashMap<u32, NodeIndex> = entries
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
            let map: FxHashMap<Value, NodeIndex> = entries.into_iter().collect();
            warn_on_duplicate_ids(node_type, entry_count, map.len());
            TypeIdIndex::General(map)
        }
    }

    /// Look up a node by type and ID value. Delegates to
    /// [`Self::lookup_by_id_normalized`], which self-heals the index on a
    /// miss, so the `&mut` buys nothing here.
    pub fn lookup_by_id(&mut self, node_type: &str, id: &Value) -> Option<NodeIndex> {
        self.lookup_by_id_normalized(node_type, id)
    }

    /// `&self` counterpart of [`Self::lookup_by_id`]; both delegate to
    /// [`Self::lookup_by_id_normalized`].
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

    pub fn has_connection_type(&self, connection_type: &str) -> bool {
        if !self.connection_types.is_empty() {
            return self
                .connection_types
                .contains(&InternedKey::from_str(connection_type));
        }
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
        // Disk-side fall-through: when the in-memory metadata looks
        // complete-but-stale (Cypher DETACH DELETE clears the
        // `connection_types` set but leaves `connection_type_metadata`
        // alone), the disk backend's `conn_type_index_*` mmap arrays are
        // authoritative for the live edge set. O(1) on disk: `Some(non-empty)`
        // iff the conn type has at least one live edge. 0.8.16.
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
        // this one): load a disk graph, add_connections of any new edge
        // type, and every subsequent typed-anchored MATCH query on an
        // existing edge type returns 0 rows.
        if self.connection_types.is_empty() && !self.connection_type_metadata.is_empty() {
            self.build_connection_types_cache();
        }
        let key = self.interner.get_or_intern(&connection_type);
        self.connection_types.insert(key);
    }

    /// Build the connection types cache. Called after deserialization or when
    /// the cache is needed.
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

        // Fallback: scan all edges (pre-metadata graphs). On the disk
        // backend `edge_weights()` materializes into the query arena,
        // which must run under a DiskQueryGuard (arena protocol in
        // disk/graph.rs, enforced by a debug assert).
        let _guard = self.graph.begin_query();
        for edge in self.graph.edge_weights() {
            self.connection_types.insert(edge.connection_type);
        }
    }

    pub fn get_type_connectivity(&self) -> Option<Vec<ConnectivityTriple>> {
        self.type_connectivity_cache.read().unwrap().clone()
    }

    pub fn set_type_connectivity(&self, triples: Vec<ConnectivityTriple>) {
        *self.type_connectivity_cache.write().unwrap() = Some(triples);
    }

    /// Get (or compute) the label-pair edge-count triples — the
    /// `(src_type, edge_type, tgt_type) → count` cardinality cache
    /// used by the Cypher planner for selectivity-aware cost estimation.
    ///
    /// Lazy: an O(E) walk on a cold cache, an O(triples) clone (typically <100
    /// entries) on a hit. Same shape as the n-triples loader's
    /// `set_type_connectivity(...)` output, so consumers can treat both as
    /// authoritative. Invalidated alongside `edge_type_counts_cache` on every
    /// edge mutation.
    pub fn get_or_compute_type_connectivity(&self) -> Vec<ConnectivityTriple> {
        {
            let read = self.type_connectivity_cache.read().unwrap();
            if let Some(ref cached) = *read {
                return cached.clone();
            }
        }
        // Arena guard: node_weight materializes on the disk backend
        // (protocol in disk/graph.rs); no-op on memory/mapped.
        let _guard = self.graph.begin_query();
        let mut counts: HashMap<(InternedKey, InternedKey, InternedKey), usize> = HashMap::new();
        for (src_idx, tgt_idx, conn_key) in self.graph.edge_endpoint_keys() {
            let src_type = match self.graph.node_type_of(src_idx) {
                Some(t) => t,
                None => continue,
            };
            let tgt_type = match self.graph.node_type_of(tgt_idx) {
                Some(t) => t,
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
    /// exact gate.
    ///
    /// **O(#types), so call it only where it is used** — roughly 23 ns per
    /// declared node type, i.e. 4.6 µs per statement on a 200-type schema if
    /// it is evaluated on every statement the planner touches rather than on
    /// the count-by-type shape that reads it. A new caller belongs behind its
    /// own shape gate.
    pub fn has_type_shadowing_property(&self) -> bool {
        self.node_type_metadata.values().any(|props| {
            props.contains_key("type")
                || props.contains_key("node_type")
                || props.contains_key("label")
        })
    }

    /// Upsert node type metadata — merges new property types into existing.
    ///
    /// **Checks before it writes**, and that check is load-bearing rather than
    /// a micro-optimisation: `node_type_metadata` is `Arc`-shared with the
    /// rollback shell (see `schema_cow`), so taking `&mut` forks the whole
    /// catalogue whether or not anything changes. The Cypher `SET` path calls
    /// this once per written row with a property the type almost always
    /// already declares, so an unconditional `&mut` would put the
    /// O(types x properties) copy back on every mutating statement.
    ///
    /// A type that is absent still falls through, so declaring a type with no
    /// properties creates its (empty) entry exactly as before.
    pub fn upsert_node_type_metadata(&mut self, node_type: &str, props: HashMap<String, String>) {
        if let Some(existing) = self.node_type_metadata.get(node_type) {
            if props.iter().all(|(k, v)| existing.get(k) == Some(v)) {
                return;
            }
        }
        let entry = self
            .node_type_metadata_mut()
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
        // Same read-first discipline as `upsert_node_type_metadata`, and for
        // the same reason: this map is `Arc`-shared with the rollback shell,
        // and the edge write path calls this per written edge.
        if let Some(existing) = self.connection_type_metadata.get(conn_type) {
            if existing.source_types.contains(source_type)
                && existing.target_types.contains(target_type)
                && prop_types
                    .iter()
                    .all(|(k, v)| existing.property_types.get(k) == Some(v))
            {
                return;
            }
        }
        let entry = self
            .connection_type_metadata_mut()
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

        for node_type in self.type_indices.keys() {
            types.insert(node_type.to_string());
        }

        // Types that have metadata but no live nodes.
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

    /// Hold the disk materialization arenas for the lifetime of a direct
    /// `GraphRead` traversal. `None` on memory/mapped backends (they don't
    /// materialize through shared arenas). Every reader that borrows node or
    /// edge weights outside the Cypher executor / pattern matcher — bindings
    /// iterating the graph directly, index builders, exporters — must keep
    /// the returned guard alive while those borrows live; see the arena
    /// SAFETY protocol in `storage/disk/graph.rs`.
    pub fn begin_read_pass(&self) -> Option<crate::graph::storage::disk::graph::DiskQueryGuard> {
        self.graph.begin_query()
    }

    /// The raw stored node record — **topology and existence only**.
    ///
    /// What the returned [`NodeData`]'s [`id()`](NodeData::id) /
    /// [`title()`](NodeData::title) hold is not the same on every backend, and
    /// that asymmetry is deliberate:
    ///
    /// * **memory / mapped** — the inline fields are the `Value::Null`
    ///   sentinel. Every ingest path is columnar, so the node's identity
    ///   lives in its type's `ColumnStore`, not on the record.
    /// * **disk** — the arena copy is materialised with the real `id`/`title`
    ///   before it is handed out, so the same call answers with real values.
    ///
    /// So use this to ask *whether* a node exists at `index`, to reach its
    /// `node_type`, or to walk topology. For any **value** read — id, title or
    /// a property — go through [`GraphRead::node_view`] (or
    /// [`GraphRead::get_node_id`]), which resolve the store and answer
    /// identically on all three backends.
    ///
    /// [`GraphRead::node_view`]: crate::graph::storage::GraphRead::node_view
    /// [`GraphRead::get_node_id`]: crate::graph::storage::GraphRead::get_node_id
    pub fn get_node(&self, index: NodeIndex) -> Option<&NodeData> {
        self.graph.node_weight(index)
    }

    // ── Column stores: DirGraph is the access point, the backend is the owner ──
    //
    // The per-type `ColumnStore` map lives on the storage backend: there is no
    // `DirGraph.column_stores` field and no mirror keeping two copies in step.
    // DirGraph keeps every lifecycle entry point — `enable_columnar`, `save`,
    // spill, vacuum — and reaches the stores through these delegates, which
    // translate the type *name* callers use into the `InternedKey` the backend
    // keys by.

    /// The store for `node_type`, if that type is columnar.
    #[inline]
    pub fn column_store(&self, node_type: &str) -> Option<&Arc<ColumnStore>> {
        self.graph.column_store(InternedKey::from_str(node_type))
    }

    /// Mutable access to `node_type`'s store, for a copy-on-write master write.
    #[inline]
    pub fn column_store_mut(&mut self, node_type: &str) -> Option<&mut Arc<ColumnStore>> {
        self.graph
            .column_store_mut(InternedKey::from_str(node_type))
    }

    #[inline]
    pub fn install_column_store(&mut self, node_type: &str, store: Arc<ColumnStore>) {
        self.graph
            .install_column_store(InternedKey::from_str(node_type), store);
    }

    #[inline]
    pub fn take_column_store(&mut self, node_type: &str) -> Option<Arc<ColumnStore>> {
        self.graph
            .take_column_store(InternedKey::from_str(node_type))
    }

    #[inline]
    pub fn clear_column_stores(&mut self) {
        self.graph.clear_column_stores();
    }

    /// Every `(type name, store)` pair, for the save / spill / vacuum paths
    /// that work in type names. O(types).
    pub fn column_stores_by_name(&self) -> Vec<(&str, &Arc<ColumnStore>)> {
        self.graph
            .column_stores_iter()
            .filter_map(|(key, store)| self.interner.try_resolve(key).map(|name| (name, store)))
            .collect()
    }

    #[inline]
    pub fn column_store_count(&self) -> usize {
        self.graph.column_stores_iter().count()
    }

    /// The authoritative read route for a node's properties — delegates to the
    /// storage backend, which resolves the node's column store.
    ///
    /// Prefer this to [`DirGraph::get_node`] for any property read. Reaching
    /// into `NodeData` reads one replica of a columnar type's store; a
    /// `NodeView` reads the one the backend answers with. See
    /// `storage/node_view.rs`.
    #[inline]
    pub fn node_view(&self, index: NodeIndex) -> Option<crate::graph::storage::NodeView<'_>> {
        self.graph.node_view(index)
    }

    /// Set one property on a node by string key — the one-call embedder
    /// route replacing the removed `NodeData::set_property`. Registers the
    /// key in the interner and routes the write by storage variant.
    ///
    /// Prefer this over calling `GraphWrite::set_node_property` on the
    /// backend directly: that trait method takes an [`InternedKey`], and a
    /// key built with `InternedKey::from_str` (which does **not** register)
    /// reads back in-session but resolves to nothing in enumerations,
    /// panics `StringInterner::resolve`, and is dropped by `save_graph` —
    /// silent data loss. Returns `false` when no node exists at `index`.
    pub fn set_node_property(&mut self, index: NodeIndex, key: &str, value: Value) -> bool {
        use crate::graph::storage::{GraphRead, GraphWrite};
        if self.graph.node_weight(index).is_none() {
            return false;
        }
        let interned = self.interner.get_or_intern(key);
        self.graph.set_node_property(index, interned, value);
        true
    }

    /// Remove one property from a node by string key — the one-call
    /// embedder route replacing the removed `NodeData::remove_property`.
    /// Routes by storage variant; returns the removed value, or `None` if
    /// the node or the property was absent. Lookup uses the pure key hash,
    /// so a never-registered key simply returns `None`.
    pub fn remove_node_property(&mut self, index: NodeIndex, key: &str) -> Option<Value> {
        use crate::graph::storage::interner::InternedKey;
        use crate::graph::storage::GraphWrite;
        self.graph
            .remove_node_property(index, InternedKey::from_str(key))
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

    /// Snapshot which property/composite/range indexes exist so they survive
    /// serialization. Called automatically before save.
    ///
    /// Every list is sorted. Each source is a `HashMap` whose iteration order is
    /// reseeded per process, and these lists are read back as sets
    /// (`rebuild_indices_from_keys`, `rebuild_unique_indices_from_keys`), so the
    /// order carries no meaning — imposing one is what makes two saves of the
    /// same graph byte-identical.
    pub fn populate_index_keys(&mut self) {
        self.property_index_keys = sorted_keys(&self.property_indices);
        self.composite_index_keys = sorted_keys(&self.composite_indices);
        self.range_index_keys = sorted_keys(&self.range_indices);
        // Declared UNIQUE constraints persist the same way. `unique_indices`
        // keys *are* the declaration list, so snapshotting them keeps the two
        // from drifting when a constraint is dropped.
        self.unique_constraint_keys = sorted_keys(&self.unique_indices);
        // Constraint *names* cannot be re-derived from the enforcement
        // structures, so unlike the lists above they are maintained live. Prune
        // instead: a name whose declaration is gone must not be saved, or
        // `DROP CONSTRAINT <name>` would resurrect it after a reload.
        self.prune_constraint_names();
    }

    /// Rebuild property, composite and range indexes from the persisted key
    /// lists. Called automatically after load.
    ///
    /// Unique constraints are rebuilt too. Any violation the loaded data already
    /// contains is discarded here rather than failing the load — see
    /// [`Self::rebuild_unique_indices_from_keys`] for why a `.kgl` must always
    /// open, and use [`Self::verify_unique_constraints`] to audit on demand.
    pub fn rebuild_indices_from_keys(&mut self) {
        let prop_keys: Vec<IndexKey> = std::mem::take(&mut self.property_index_keys);
        for (node_type, property) in &prop_keys {
            self.create_index(node_type, property);
        }
        self.property_index_keys = prop_keys;

        // A `.kgl` written before composite keys were canonicalized carries the
        // declaration order; `create_composite_index` sorts it, so re-deriving
        // the list from the rebuilt map keeps the snapshot and the live index
        // agreeing on one spelling instead of persisting the old one again.
        let comp_keys: Vec<CompositeIndexKey> = std::mem::take(&mut self.composite_index_keys);
        for (node_type, properties) in &comp_keys {
            let prop_refs: Vec<&str> = properties.iter().map(|s| s.as_str()).collect();
            self.create_composite_index(node_type, &prop_refs);
        }
        self.composite_index_keys = sorted_keys(&self.composite_indices);

        let range_keys: Vec<IndexKey> = std::mem::take(&mut self.range_index_keys);
        for (node_type, property) in &range_keys {
            self.create_range_index(node_type, property);
        }
        self.range_index_keys = range_keys;

        let _preexisting_violations = self.rebuild_unique_indices_from_keys();
    }

    /// Rebuild type_indices from the live graph. Called after deserialization
    /// (type_indices is `#[serde(skip)]`) and by [`Self::reindex`].
    pub fn rebuild_type_indices(&mut self) {
        let grouped = self.group_node_indices_by_type(self.node_type_metadata.len());
        self.type_indices.replace_with(grouped);
        // `secondary_label_index` is *not* rebuilt from node data — it's
        // the canonical store, populated either by the choke-point API
        // during the session or by the load path (the disk sidecar /
        // the in-memory .kgl section).
    }

    /// One pass over the nodes producing `type name -> node indices`.
    /// `type_hint` is the caller's expectation of how many distinct node types
    /// there are, used only to size the map and the per-type vectors.
    ///
    /// Group on the node's *interned* type key, not its name. The name is a
    /// per-type fact but this loop is per-node: resolving it and allocating a
    /// `String` for every node — then hashing that string with SipHash on the
    /// way into the map — was ~10% of a fired vacuum at 1M. `InternedKey` is a
    /// `Copy` integer, so grouping on it costs nothing and the O(types)
    /// resolve happens once, at the end.
    fn group_node_indices_by_type(&self, type_hint: usize) -> HashMap<String, Vec<NodeIndex>> {
        let type_count = type_hint.max(4);
        let avg_per_type = self.graph.node_count() / type_count.max(1);
        let mut by_type_key: FxHashMap<InternedKey, Vec<NodeIndex>> =
            FxHashMap::with_capacity_and_hasher(type_count, Default::default());
        {
            // Arena guard: node_weight materializes on the disk backend
            // (protocol in disk/graph.rs); scoped so the borrow ends with the
            // scan.
            let _guard = self.graph.begin_query();
            for node_idx in self.graph.node_indices() {
                if let Some(node) = self.graph.node_view(node_idx) {
                    by_type_key
                        .entry(node.node_type())
                        .or_insert_with(|| Vec::with_capacity(avg_per_type))
                        .push(node_idx);
                }
            }
        }
        let mut by_type_name: HashMap<String, Vec<NodeIndex>> =
            HashMap::with_capacity(by_type_key.len());
        for (type_key, indices) in by_type_key {
            by_type_name.insert(self.interner.resolve(type_key).to_string(), indices);
        }
        by_type_name
    }

    /// Rebuild `self.type_schemas` — one shared [`TypeSchema`] per node type,
    /// derived from `node_type_metadata` (O(types)) or, for a graph that
    /// carries none, by scanning the staged property maps. Read by the column
    /// store constructors and by the planner's type metadata.
    pub fn rebuild_type_schemas(&mut self) {
        let mut schemas: HashMap<String, TypeSchema> = HashMap::new();
        for (node_type, props) in self.node_type_metadata.iter() {
            // SLOT-ORDER RULE: sorted by property name.
            //
            // `props` is a `HashMap`, so iterating it directly made a type's
            // slot order a per-process `RandomState` artefact. Slot order is
            // observable — it is the order columns are written into a `.kgl`
            // payload — so a random order meant re-saving a loaded graph
            // produced different bytes on every run.
            //
            // Nothing here records an intended order (unlike a packed column
            // payload, which carries one positionally and is honoured by
            // `io::file::load_portable_column_section`), so sorted-by-name is
            // the canonical choice: deterministic, independent of insertion
            // history, and stable across processes and platforms.
            let mut names: Vec<&String> = props.keys().collect();
            names.sort();
            let keys = names
                .into_iter()
                .map(|name| self.interner.get_or_intern(name));
            schemas.insert(node_type.clone(), TypeSchema::from_keys(keys));
        }

        // Fallback: if metadata is empty (pre-metadata graph), scan nodes.
        // Arena guard: node_weight materializes on the disk backend
        // (protocol in disk/graph.rs); scoped so the borrow ends with the
        // fallback scan rather than outliving it.
        if schemas.is_empty() {
            let _guard = self.graph.begin_query();
            for node_idx in self.graph.node_indices() {
                if let Some(node) = self.graph.node_weight(node_idx) {
                    let type_str = node.node_type_str(&self.interner).to_string();
                    let schema = schemas.entry(type_str).or_insert_with(TypeSchema::new);
                    if let PropertyStorage::Map(map) = &node.properties {
                        // Same slot-order rule: `map` is a `HashMap`, so sort
                        // before adding. Sorting the interned keys (a stable
                        // FNV `u64` of the name) is deterministic and avoids
                        // resolving every key back to a string in this
                        // per-node loop.
                        let mut keys: Vec<InternedKey> = map.keys().copied().collect();
                        keys.sort();
                        for key in keys {
                            schema.add_key(key);
                        }
                    }
                }
            }
        }

        self.type_schemas = Arc::new(schemas.into_iter().map(|(t, s)| (t, Arc::new(s))).collect());
    }

    /// Combined [`Self::rebuild_type_indices`] + [`Self::rebuild_type_schemas`],
    /// used after deserialization where both need to run. Schemas come from
    /// `node_type_metadata` (O(types)), so the usual cost is the one node pass
    /// the type-index grouping needs; only a graph carrying no metadata pays a
    /// second pass for the schema fallback scan.
    pub fn rebuild_type_indices_and_schemas(&mut self) {
        let mut schemas: HashMap<String, TypeSchema> = HashMap::new();
        for (node_type, props) in self.node_type_metadata.iter() {
            // SLOT-ORDER RULE: sorted by property name — see
            // `rebuild_type_schemas` for why the order is observable.
            let mut names: Vec<&String> = props.keys().collect();
            names.sort();
            let keys = names
                .into_iter()
                .map(|name| self.interner.get_or_intern(name));
            schemas.insert(node_type.clone(), TypeSchema::from_keys(keys));
        }

        // Fallback: if metadata is empty (loaded from file), scan nodes.
        // Arena guard: node_weight materializes on the disk backend
        // (protocol in disk/graph.rs); scoped so the borrow ends before
        // the type-index grouping pass below takes its own.
        if schemas.is_empty() {
            let _guard = self.graph.begin_query();
            for node_idx in self.graph.node_indices() {
                if let Some(node) = self.graph.node_weight(node_idx) {
                    let type_str = node.node_type_str(&self.interner).to_string();
                    let schema = schemas.entry(type_str).or_insert_with(TypeSchema::new);
                    if let PropertyStorage::Map(map) = &node.properties {
                        // Same slot-order rule; sorting the interned keys (a
                        // stable FNV `u64`) avoids resolving every key back to
                        // a string in this per-node loop.
                        let mut keys: Vec<InternedKey> = map.keys().copied().collect();
                        keys.sort();
                        for key in keys {
                            schema.add_key(key);
                        }
                    }
                }
            }
        }

        let arc_schemas: HashMap<String, Arc<TypeSchema>> =
            schemas.into_iter().map(|(t, s)| (t, Arc::new(s))).collect();

        let new_type_indices = self.group_node_indices_by_type(arc_schemas.len());

        self.type_indices.replace_with(new_type_indices);
        self.type_schemas = Arc::new(arc_schemas);
        // `secondary_label_index` is *not* rebuilt here — it's the
        // canonical store, populated by the load path (the disk
        // sidecar or the in-memory `.kgl` section).
    }

    /// Ensure a ColumnStore exists for `node_type`, creating an empty one on
    /// the type's registered `TypeSchema` if it does not. Call
    /// `ensure_type_schema_keys()` first to register new keys.
    ///
    /// **Schema growth does not rebuild the store.** A key the store's own
    /// schema has never seen appends one column inside
    /// [`ColumnStore::push_row`](crate::graph::storage::column_store::ColumnStore::push_row),
    /// back-filled with nulls — O(rows) once per column, amortized O(1) per
    /// row. This used to re-push every existing row into a fresh store on
    /// every newly-seen key instead, which made a stream that widens its key
    /// set quadratic in the rows already present, and dropped the tombstone
    /// bitmap on the way across so a row deleted before the growth came back
    /// to life. The store's schema may therefore lag or re-order the
    /// registered `TypeSchema`; nothing indexes one by the other's slots
    /// (columnar rows carry a row id, and every read resolves its slot through
    /// the store's own schema).
    pub fn ensure_column_store_for_push(
        &mut self,
        node_type: &str,
    ) -> &mut crate::graph::storage::column_store::ColumnStore {
        use crate::graph::storage::column_store::ColumnStore;

        if self.column_store(node_type).is_none() {
            let schema = self
                .type_schemas
                .get(node_type)
                .cloned()
                .unwrap_or_else(|| Arc::new(TypeSchema::new()));
            let meta = self
                .node_type_metadata
                .get(node_type)
                .cloned()
                .unwrap_or_default();
            let store = ColumnStore::new(schema, &meta, &self.interner);
            self.install_column_store(node_type, Arc::new(store));
        }

        // An mmap-backed store must become owned before a row is appended.
        // `push_id`/`push_title` create their overlay columns at row zero, so
        // alongside a live mmap base every appended id/title lands `row_count`
        // rows too early — the overlay then shadows the mapped originals on
        // every read, and a save serializes a title column shorter than the
        // rows it advertises. O(rows) once per store: afterwards there is no
        // mmap base to check.
        if self
            .column_store(node_type)
            .is_some_and(|store| store.has_mmap_base())
        {
            let meta = self
                .node_type_metadata
                .get(node_type)
                .cloned()
                .unwrap_or_default();
            if let Some(arc) = self.take_column_store(node_type) {
                let mut store = Arc::try_unwrap(arc).unwrap_or_else(|a| (*a).clone());
                store.materialize_for_append(&meta, &self.interner);
                self.install_column_store(node_type, Arc::new(store));
            }
        }

        Arc::make_mut(self.column_store_mut(node_type).unwrap())
    }

    /// Ensure the TypeSchema for `node_type` contains all the given keys.
    /// Creates the schema if it doesn't exist, extends it if it does.
    ///
    /// Read-first, like the two metadata upserts above: `type_schemas` is
    /// `Arc`-shared with the rollback shell, and the create path calls this
    /// once per created node with a key set the type already carries.
    pub fn ensure_type_schema_keys(&mut self, node_type: &str, keys: &[InternedKey]) {
        if let Some(schema) = self.type_schemas.get(node_type) {
            if keys.iter().all(|key| schema.slot(*key).is_some()) {
                return;
            }
        }
        let schema = self
            .type_schemas_mut()
            .entry(node_type.to_string())
            .or_insert_with(|| Arc::new(TypeSchema::new()));
        let s = Arc::make_mut(schema);
        for &key in keys {
            s.add_key(key);
        }
    }

    /// Check heap usage of column stores and spill largest to disk if over
    /// limit. No-op if `memory_limit` is None.
    ///
    /// Two guards decide, in order, that there is nothing to do — and both are
    /// load-bearing, because this runs after *every* mutating statement
    /// (`session::execute`):
    ///
    /// 1. **No store grew spillable heap since its last spill.** A `SET` of an
    ///    existing property on a mapped column is written through the mapping
    ///    and adds nothing; walking every type to rediscover that is pure
    ///    per-statement tax, O(types) wide.
    /// 2. **The spillable total is under the limit.** The comparison is against
    ///    [`ColumnStore::spillable_heap_bytes`], not `heap_bytes`: the latter
    ///    includes the tombstone bitmap, the `Str` `relocated` overlay, `Mixed`
    ///    columns and the overflow bag, none of which `materialize_to_files`
    ///    can move. `StorageMode::Mapped` pins `memory_limit = Some(0)`, so
    ///    that floor kept the total permanently over the limit: the pass could
    ///    never converge and re-ran its whole per-type loop — Vec + sort + a
    ///    `create_dir_all` syscall per type — on every statement, spilling
    ///    nothing. Measured at 100 types: 245 µs per single-row `SET`.
    ///
    /// `graph_info()['columnar_heap_bytes']` keeps reporting the full
    /// `heap_bytes` — the floor is excluded from the *decision*, not from the
    /// observability reading.
    pub fn maybe_spill_columns(&mut self) {
        let limit = match self.memory_limit {
            Some(l) => l,
            None => return,
        };
        if !self
            .graph
            .column_stores_iter()
            .any(|(_, s)| s.may_have_grown_spillable_heap())
        {
            return;
        }
        let total: usize = self
            .graph
            .column_stores_iter()
            .map(|(_, s)| s.spillable_heap_bytes())
            .sum();
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
        if self.spill_dir.is_none() {
            self.spill_dir = Some(spill_dir.clone());
        }
        // Register for cleanup on drop
        if let Ok(mut dirs) = self.temp_dirs.lock() {
            if !dirs.contains(&spill_dir) {
                dirs.push(spill_dir.clone());
            }
        }

        // Spill largest stores first until under limit.
        let mut by_size: Vec<(String, usize)> = self
            .column_stores_by_name()
            .into_iter()
            .map(|(t, s)| (t.to_string(), s.spillable_heap_bytes()))
            .collect();
        by_size.sort_by_key(|s| std::cmp::Reverse(s.1));
        let mut remaining = total;
        for (type_name, bytes) in by_size {
            if remaining <= limit {
                break;
            }
            let type_dir = spill_dir.join(&type_name);
            // The backend owns the store and nothing else holds a handle, so
            // `make_mut` mutates it in place: the spill actually reclaims the
            // heap it materialises to files. Back when every node also held a
            // strong `Arc` on the store, this forked and reclaimed nothing.
            let interner = &self.interner;
            let Some(arc) = self
                .graph
                .column_store_mut(InternedKey::from_str(&type_name))
            else {
                continue;
            };
            if Arc::make_mut(arc)
                .materialize_to_files(&type_dir, interner)
                .is_ok()
            {
                remaining -= bytes;
            }
        }
    }

    pub fn reindex(&mut self) {
        self.rebuild_type_indices();

        // Lazy caches rebuild on next access.
        self.id_indices.clear();
        self.connection_types.clear();

        // Rebuild each index that exists, preserving *which* ones exist.
        let property_keys: Vec<IndexKey> = self.property_indices.keys().cloned().collect();
        for (node_type, property) in property_keys {
            self.create_index(&node_type, &property);
        }

        let composite_keys: Vec<CompositeIndexKey> =
            self.composite_indices.keys().cloned().collect();
        for (node_type, properties) in composite_keys {
            let prop_refs: Vec<&str> = properties.iter().map(|s| s.as_str()).collect();
            self.create_composite_index(&node_type, &prop_refs);
        }

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
    /// Returns a [`NodeRemap`] from old NodeIndex → new NodeIndex so callers
    /// can update any external references (e.g., selections). An empty mapping
    /// means nothing was remapped.
    ///
    /// Skips the rebuild when there is no petgraph tombstone of either shape
    /// (`node_count == node_bound` *and* `edge_count == edge_bound`); column
    /// stores still holding orphaned rows are rebuilt even then.
    ///
    /// A full no-op on the **disk** backend: its CSR arrays are frozen mmap,
    /// not a `StableDiGraph`, so there is no petgraph tombstone to compact —
    /// disk reclaims space by publishing a fresh generation, whose columns are
    /// written without the rows no live node points at
    /// ([`DirGraph::save_disk`]), not by rebuilding in place. Rebuilding would
    /// also have to materialise the whole graph on the heap, which is the one
    /// thing the disk backend exists to avoid.
    ///
    /// The rebuild preserves the backend variant and any write-capture
    /// wrapper. **Callers on a durable graph must flush the write-ahead log
    /// first**: buffered ops are keyed by `NodeIndex` and every index moves
    /// here.
    pub fn vacuum(&mut self) -> NodeRemap {
        if self.graph.is_disk() {
            return NodeRemap::default();
        }
        let old_node_count = self.graph.node_count();
        let old_node_bound = self.graph.node_bound();
        // Free *edge* slots are reclaimed by the same rebuild — it re-adds
        // every live edge into a fresh graph — but only if the rebuild runs.
        // Gating the whole pass on the node reading alone made `vacuum()` a
        // measured no-op for a relationship-only delete workload: 500 of 1,000
        // edges deleted, `tombstones_removed` 0, edge slots still held.
        let edge_tombstones = self
            .graph
            .edge_bound()
            .saturating_sub(self.graph.edge_count());

        // No petgraph tombstones of either shape — but columnar stores may
        // still have orphaned rows (e.g. all nodes deleted → petgraph is empty
        // but column data remains).
        if old_node_count == old_node_bound && edge_tombstones == 0 {
            let columnar_orphaned = self.column_stores_by_name().into_iter().any(|(t, s)| {
                let live = self.type_indices.get(t).map(|v| v.len()).unwrap_or(0);
                (s.row_count() as usize) > live
            });
            if columnar_orphaned {
                self.rebuild_columns_to_heap();
            }
            return NodeRemap::default();
        }

        // Take ownership of the old graph so the rebuild can *relocate* every
        // weight instead of deep-cloning it. A `NodeData` clone reallocates
        // the id/title strings and the whole property vector, and the
        // originals were dropped moments later when the backend was replaced
        // — the clone/drop pair was pure waste (profiled at ~16% of a fired
        // vacuum at 1M, plus its share of allocator and memcpy time).
        let Some(mut old) = self.graph.take_heap_graph() else {
            // Disk: nothing was taken, so nothing downstream may treat the
            // indices as remapped.
            return NodeRemap::default();
        };

        let mut new_graph = StableDiGraph::with_capacity(old_node_count, old.edge_count());
        // Dense old→new lookup, for the edge pass *and* for the caller.
        // Endpoint remapping is two probes per edge, and running them through a
        // `HashMap`'s SipHash was the largest single cost in the rebuild
        // (hashing was ~22% of a fired vacuum at 1M). The graph is
        // index-addressed, so a bound-sized vector is the natural map — and it
        // is the *only* one now: the second, hash-based copy this function used
        // to build for its return value carried no information this one lacks
        // (`NodeRemap`).
        let mut old_to_new = NodeRemap::with_bound(old_node_bound);

        // Ascending raw order reproduces `node_indices()` exactly, so the
        // compacted indices are the same ones the clone loop produced.
        for raw in 0..old_node_bound {
            let old_idx = NodeIndex::new(raw);
            let Some(slot) = old.node_weight_mut(old_idx) else {
                continue;
            };
            let vacated = NodeData {
                id: Value::Null,
                title: Value::Null,
                node_type: slot.node_type,
                // `HashMap::new` does not allocate, so the placeholder left
                // in the discarded graph costs nothing to build or drop.
                properties: PropertyStorage::Map(HashMap::new()),
            };
            let node_data = std::mem::replace(slot, vacated);
            let new_idx = new_graph.add_node(node_data);
            old_to_new.set(raw, new_idx);
        }

        // The edge ids are collected because relocating a weight needs
        // `&mut old` while `edge_indices()` borrows it.
        let old_edge_ids: Vec<EdgeIndex> = old.edge_indices().collect();
        for old_edge_idx in old_edge_ids {
            let Some((src, tgt)) = old.edge_endpoints(old_edge_idx) else {
                continue;
            };
            let (new_src, new_tgt) = (old_to_new.raw(src.index()), old_to_new.raw(tgt.index()));
            debug_assert!(
                !NodeRemap::is_vacant(new_src) && !NodeRemap::is_vacant(new_tgt),
                "a live edge referenced a node that was not carried over"
            );
            let Some(slot) = old.edge_weight_mut(old_edge_idx) else {
                continue;
            };
            let vacated = EdgeData {
                connection_type: slot.connection_type,
                properties: Vec::new(),
            };
            let edge_data = std::mem::replace(slot, vacated);
            new_graph.add_edge(
                NodeIndex::new(new_src as usize),
                NodeIndex::new(new_tgt as usize),
                edge_data,
            );
        }
        drop(old);

        // Replace graph storage, keeping the backend variant and any
        // write-capture wrapper — see `GraphBackend::replace_heap_graph` for
        // what assigning `Memory(..)` here used to break.
        if !self.graph.replace_heap_graph(new_graph) {
            // Unreachable: `take_heap_graph` already returned `None` for the
            // only backend `replace_heap_graph` refuses.
            return NodeRemap::default();
        }

        self.remap_embedding_slots(&old_to_new);
        // Secondary labels live above the backend (labels.rs module doc), so
        // neither the clone loop nor reindex() below can carry them over.
        self.remap_secondary_labels(&old_to_new);
        // Text indexes are dropped, not remapped. A document's slot *is* its
        // node index (`graph::text_indexes`), so after a wholesale remap every
        // document would be attached to whatever node inherited its number —
        // and rebuilding them here would hide a full corpus re-index inside a
        // maintenance call. Same call the HNSW index gets one line above, for
        // the same reason: rebuild after a vacuum.
        self.text_indexes.clear();
        self.reindex();

        // Rebuild the columnar stores: the old ones carry orphaned rows from
        // deleted nodes, and every row id the compaction just invalidated. The
        // disable/enable cycle reads only live nodes, producing fresh
        // `ColumnStore`s with no dead rows.
        //
        // Still gated, and the gate still has a false arm — narrowly. Every
        // *construction* funnel is columnar now (Cypher `CREATE`, the bulk
        // batch path, the loaders), so this is true for any graph a user
        // built; what can still own no store is a graph assembled by a direct
        // `GraphWrite::add_node` (the RDF loader, and the Rust tests that build
        // fixtures that way), whose nodes hold their properties inline.
        // Converting one here would be a shape change, not a compaction, so
        // the gate stays.
        if self.column_store_count() > 0 {
            self.rebuild_columns_to_heap();
        }

        old_to_new
    }

    /// Rebuild every column store from live nodes, heap-resident, with the
    /// memory limit taken out of the way for the duration.
    ///
    /// The primitive behind `vacuum`'s columnar half and the public
    /// `unspill()`. Both want the same two effects and neither wants a row
    /// shape on the way: rows that no live node points at are dropped (the
    /// rebuild reads live nodes only), and every column comes back as a fresh
    /// heap column, which is what un-spills an mmap-backed store.
    ///
    /// The limit is suspended because the rebuild goes through the heap by
    /// construction — leaving it installed would re-spill each store as the
    /// rebuild passed it, only for the next statement's write to pull it back.
    /// Restoring it afterwards leaves the graph heap-resident, which is the
    /// documented behaviour (`test_vacuum_columnar_with_memory_limit`,
    /// `test_unspill_preserves_memory_limit`).
    ///
    /// This replaces the de-columnarize + re-consolidate round trip the two
    /// callers used to make, which materialised every row onto its node only
    /// for the very next pass to read it back off again.
    pub fn rebuild_columns_to_heap(&mut self) {
        let saved_limit = self.memory_limit.take();
        self.rebuild_column_stores();
        self.memory_limit = saved_limit;
    }

    /// Rows held by every per-type `ColumnStore`, and how many of them a live
    /// node still points at: `(total, live)`.
    ///
    /// The columnar half of the fragmentation picture, and the only one that
    /// sees garbage produced by *replacement*. Shared by
    /// [`Self::graph_info`] and [`Self::check_auto_vacuum`] so the number a
    /// caller reads and the number the trigger acts on cannot drift apart.
    ///
    /// `live` counts nodes of the type, not columnar rows specifically. That
    /// used to under-report garbage on a graph whose types carried a mix of
    /// columnar and row-shaped nodes; construction is columnar, so the two
    /// counts describe the same population and the census is exact. The
    /// saturating subtraction every consumer does is kept for the transient
    /// shapes that still exist (a `Map`-staged node mid-ingest), where it fails
    /// towards *not* vacuuming.
    pub(crate) fn columnar_row_census(&self) -> (usize, usize) {
        let total = self
            .graph
            .column_stores_iter()
            .map(|(_, s)| s.row_count() as usize)
            .sum();
        let live = self
            .column_stores_by_name()
            .into_iter()
            .map(|(t, _)| self.type_indices.get(t).map(|v| v.len()).unwrap_or(0))
            .sum();
        (total, live)
    }

    /// Check whether auto-vacuum should run after a delete, and run it if so.
    ///
    /// Called after `DELETE` / `DETACH DELETE`. Vacuums only when
    /// `auto_vacuum_threshold` is set, the largest of the three garbage
    /// populations (free node slots, dead columnar rows, free edge slots)
    /// exceeds 100 — so tiny graphs never pay for a rebuild — and the worst of
    /// their three fragmentation ratios exceeds the threshold.
    ///
    /// # Return value
    ///
    /// `None` when no vacuum ran. `Some(remap)` when one did, carrying the
    /// `old → new` node mapping so a caller holding node indices (a fluent
    /// selection, an external cursor) can *follow* the compaction instead of
    /// throwing its state away.
    ///
    /// **This used to be a `bool`, and the `bool` was a footgun**: on the disk
    /// backend `vacuum()` is a no-op — its CSR arrays are frozen mmap, so
    /// there is no petgraph tombstone to compact — yet the trigger still
    /// returned `true`, and its one caller read that as "indices moved" and
    /// reset the caller's selection. Disk paid the whole cost of a compaction
    /// while reclaiming nothing. A returned remap cannot lie that way: the
    /// no-op hands back a mapping that
    /// [`describes_rebuild`](NodeRemap::describes_rebuild) reports as no
    /// rebuild, which is exactly the fact the caller needs.
    pub fn check_auto_vacuum(&mut self) -> Option<NodeRemap> {
        let threshold = self.auto_vacuum_threshold?;

        let node_count = self.graph.node_count();
        let node_bound = self.graph.node_bound();
        let tombstones = node_bound - node_count;

        // Columnar garbage is a second, independent kind of fragmentation, and
        // the node-slot count cannot stand in for it. A delete leaves a dead
        // row behind in the type's store; a *later create* takes the freed
        // petgraph slot, so `node_bound - node_count` returns to zero while the
        // store keeps both the dead row and a fresh one. Measured on the disk
        // backend (already always-columnar): 1,500 delete/create pairs over a
        // 2,000-node type left 3,500 rows for 2,000 live nodes — 43% garbage —
        // at `fragmentation_ratio` 0.000. Deletes alone are tracked correctly
        // by the node count, which is why this stayed invisible until
        // replacement churn was measured.
        let (columnar_total, columnar_live) = self.columnar_row_census();
        let columnar_dead = columnar_total.saturating_sub(columnar_live);

        // Edge slots are the third, equally independent kind of garbage, and
        // until this was added they were invisible to the trigger: a workload
        // that deletes only *relationships* (`MATCH ()-[r]->() DELETE r`)
        // leaves every node alive and every columnar row referenced, so both
        // numbers above read clean. Measured before the fix: 500 of 1,000
        // edges deleted reported `fragmentation_ratio` 0.000, could never
        // auto-vacuum, and got a no-op from an explicit `vacuum()` as well.
        let edge_count = self.graph.edge_count();
        let edge_bound = self.graph.edge_bound();
        let edge_tombstones = edge_bound.saturating_sub(edge_count);

        // The small-graph floor applies to whichever kind of garbage there is.
        if tombstones.max(columnar_dead).max(edge_tombstones) <= 100 {
            return None;
        }

        let ratio = |dead: usize, bound: usize| {
            if bound == 0 {
                0.0
            } else {
                dead as f64 / bound as f64
            }
        };
        let worst = ratio(tombstones, node_bound)
            .max(ratio(columnar_dead, columnar_total))
            .max(ratio(edge_tombstones, edge_bound));
        if worst > threshold {
            self.auto_vacuums_run += 1;
            Some(self.vacuum())
        } else {
            None
        }
    }

    /// Return diagnostic information about graph storage health. A
    /// `fragmentation_ratio` above 0.3 (the auto-vacuum default) is a good
    /// point to call [`Self::vacuum`].
    pub fn graph_info(&self) -> GraphInfo {
        let (columnar_total, columnar_live) = self.columnar_row_census();
        let (edges_mapped, edge_property_overlay_rows) = self.graph.edge_storage_info();
        let node_count = self.graph.node_count();
        let node_bound = self.graph.node_bound();
        let edge_count = self.graph.edge_count();
        let edge_bound = self.graph.edge_bound();
        let node_tombstones = node_bound - node_count;

        GraphInfo {
            node_count,
            node_capacity: node_bound,
            node_tombstones,
            edge_count,
            edge_capacity: edge_bound,
            edge_tombstones: edge_bound.saturating_sub(edge_count),
            fragmentation_ratio: if node_bound == 0 {
                0.0
            } else {
                node_tombstones as f64 / node_bound as f64
            },
            type_count: self.type_indices.len(),
            property_index_count: self.property_indices.len(),
            composite_index_count: self.composite_indices.len(),
            columnar_total_rows: columnar_total,
            columnar_live_rows: columnar_live,
            columnar_heap_bytes: self
                .graph
                .column_stores_iter()
                .map(|(_, s)| s.heap_bytes())
                .sum(),
            columnar_is_mapped: self.graph.column_stores_iter().any(|(_, s)| s.is_mapped()),
            edges_mapped,
            edge_property_overlay_rows,
        }
    }
}

#[derive(Debug, Clone)]
pub struct IndexStats {
    pub unique_values: usize,
    pub total_entries: usize,
    pub avg_entries_per_value: f64,
}

/// Diagnostic information about graph storage health.
#[derive(Debug, Clone)]
pub struct GraphInfo {
    pub node_count: usize,
    /// Upper bound of node indices (includes tombstones from deletions)
    pub node_capacity: usize,
    /// Number of tombstone slots (node_capacity - node_count)
    pub node_tombstones: usize,
    pub edge_count: usize,
    /// Upper bound of edge indices (includes slots freed by `DELETE r`).
    ///
    /// Always equal to [`Self::edge_count`] on the disk backend, whose edges
    /// are a frozen CSR generation with no free list.
    pub edge_capacity: usize,
    /// Number of free edge slots (`edge_capacity - edge_count`).
    ///
    /// The third garbage population, alongside node tombstones and dead
    /// columnar rows, and the only one a relationship-only delete workload
    /// produces. Reported separately rather than folded into
    /// [`Self::fragmentation_ratio`], which stays node-shaped so its documented
    /// meaning does not silently change; the auto-vacuum trigger takes the
    /// worst of all three.
    pub edge_tombstones: usize,
    /// Ratio of wasted node storage (0.0 = clean, approaching 1.0 = heavily
    /// fragmented). Node slots only — see [`Self::edge_tombstones`] and
    /// [`Self::columnar_total_rows`] for the other two populations.
    pub fragmentation_ratio: f64,
    pub type_count: usize,
    /// Number of single-property indexes
    pub property_index_count: usize,
    pub composite_index_count: usize,
    /// Total rows across all columnar stores (including orphaned from deletions)
    pub columnar_total_rows: usize,
    /// Rows backed by live nodes (columnar_total_rows - columnar_live_rows = orphaned)
    pub columnar_live_rows: usize,
    /// Heap bytes held by the column stores the backend owns. Exposed here so
    /// a binding can report columnar memory without reaching into storage.
    pub columnar_heap_bytes: usize,
    /// `true` when at least one column store has been spilled to mmap.
    ///
    /// Reports **property-column spilling** (`set_memory_limit`, or the
    /// `mapped` storage mode which pins that limit at 0) — not disk-mode
    /// health. A disk graph reports `false` here unless its *columns* also
    /// spilled; [`Self::edges_mapped`] is the disk-mode reading.
    pub columnar_is_mapped: bool,
    /// `true` when the edge CSR arrays are memory-mapped from files rather
    /// than heap-resident — the structure `enable_disk_mode()` materializes.
    /// Always `false` on the memory and mapped backends, which keep edges in
    /// the heap graph and have no CSR at all.
    pub edges_mapped: bool,
    /// Edges whose properties sit in the disk backend's heap mutation overlay
    /// instead of the mmap-backed columnar base. `0` on every non-disk
    /// backend.
    pub edge_property_overlay_rows: usize,
}

// `make_dir_graph_mut` (the `Arc<DirGraph>` → `&mut DirGraph` + version-bump
// handle) lives in `crate::graph::handle` to keep this file under the
// god-file ceiling; it is re-exported through `kglite::api::make_dir_graph_mut`.

#[cfg(test)]
#[path = "dir_graph_tests.rs"]
mod dir_graph_tests;

#[cfg(test)]
mod rollback_tests;

#[cfg(test)]
mod fork_apportionment_tests;

#[cfg(test)]
mod disk_snapshot_tests;
