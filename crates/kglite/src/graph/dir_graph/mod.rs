//! DirGraph — transactional container for the in-memory graph.
//!
//! Owns the `StableDiGraph` + all type/property/composite/range indexes,
//! OCC `version`, `schema_locked`, spatial / temporal / timeseries configs,
//! embedding stores, connection-type metadata, and schema definitions.

use self::index_layer::LayeredIndex;
use crate::datatypes::values::Value;
use crate::graph::constraints::{NamedConstraint, UniqueConstraintKey};
use crate::graph::schema::{
    ColumnarRow, CompositeIndexKey, CompositeValue, ConnectionTypeInfo, ConnectivityTriple,
    EdgeData, EmbeddingStore, GraphBackend, IndexKey, InternedKey, NodeData, PropertyStorage,
    SaveMetadata, SchemaDefinition, SpatialConfig, StringInterner, TemporalConfig, TypeIdIndex,
    TypeSchema,
};
use crate::graph::storage::column_store::ColumnStore;
use crate::graph::storage::disk::id_index::IdIndexStore;
use crate::graph::storage::disk::type_index::TypeIndexStore;

// Counts full `enable_columnar` rebuilds, so the save fast path can be pinned
// by measurement rather than by argument (D1 risk 1). Test-only.
//
// Thread-local: `cargo test` runs tests in parallel, and a global counter would
// make the rebuild count depend on what else happened to be running.
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

/// Core graph storage: a directed graph (petgraph `StableDiGraph`) with fast
/// type-based indexing and optional property/composite/range/spatial indexes.
///
/// Fields include `type_indices` for O(1) node-type lookup, `property_indices`
/// for indexed equality filters, connection-type metadata, schema definitions,
/// and optional embedding stores for vector similarity search.
/// Source of process-unique graph ids. Starts at 1 (0 is never handed out, so
/// it can serve as a sentinel); monotonic and never reused so a dropped graph's
/// plan-cache entries can't be served to a later graph that reuses an address.
static NEXT_GRAPH_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Mint a fresh, process-unique graph id (also the serde default for the
/// skipped `graph_id` field, so a loaded graph gets a new identity).
fn next_graph_id() -> u64 {
    NEXT_GRAPH_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

// Lazily-computed caches + derived stats (edge-type counts, type
// connectivity, per-(type,property) NDV) live in a child module so the
// file stays under the god-file ceiling; child = retains private access.
pub(crate) mod caches;
pub mod constraints;
mod disk_persistence;
mod independent_copy;
pub mod index_layer;
mod indexes;
mod labels;
mod node_write;
pub(crate) mod rollback;
mod schema_ops;

/// Version-keyed cache of per-`(type, property)` distinct-value counts (NDV)
/// for the planner's selectivity estimator. The `u64` is the graph `version`
/// the map was built at; a mismatch triggers a recompute (auto-invalidation).
type PropertyNdvCache = Arc<RwLock<(u64, HashMap<(String, String), usize>)>>;

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
    ///
    /// Each index's `value -> members` map is a [`LayeredIndex`]: a stack of
    /// shared, immutable levels, so a fork shares the buckets instead of
    /// copying one `Value` key and one `Vec` per distinct value (D2 — 48.0 ms
    /// at 1M before layering). Reads and edits keep the `HashMap` shape.
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
    /// B-Tree range indexes for ordered lookups: (node_type, property) -> BTreeMap<Value, [NodeIndex]>
    /// Skipped during serialization — rebuilt from `range_index_keys` on load.
    #[serde(skip)]
    pub range_indices: HashMap<IndexKey, std::collections::BTreeMap<Value, Vec<NodeIndex>>>,
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
    /// `BTreeSet` so the persisted bytes are deterministic. Additive serde field
    /// — older `.kgl` files load with an empty set, which only means their DDL
    /// presence constraints are indistinguishable from schema-declared ones,
    /// exactly as they are today.
    #[serde(default)]
    pub(crate) ddl_not_null_constraints: std::collections::BTreeSet<(String, String)>,
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
    #[serde(default)]
    pub node_type_metadata: HashMap<String, HashMap<String, String>>,
    /// Connection type metadata: connection_type → ConnectionTypeInfo
    /// Replaces SchemaNode graph nodes for connections — persisted via versioned binary Serde.
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
    ///
    /// **Fork-private** since D2 Phase 3 — see [`caches::ForkPrivateCache`] for
    /// the aliasing bug that earned it (a snapshot reporting the writer's edge
    /// counts) and for why `wkt_cache` and `property_ndv_cache` deliberately
    /// stay shared.
    #[serde(skip)]
    pub edge_type_counts_cache: caches::ForkPrivateCache<HashMap<String, usize>>,
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
    /// Transient, **execution-scoped** write whitelist. When `Some(set)`, a
    /// Cypher `CREATE`/`SET` whose node type is not in `set` is rejected
    /// (role-scoped writes — integrity, not secrecy). Set by `execute_mut`
    /// for the duration of one mutation and cleared immediately after; never
    /// a persistent graph property and never serialized. `None` = unrestricted
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
    /// Stored **with the exact message it produced**, and the drain only
    /// accepts it when the string that arrived is byte-identical: if any
    /// intermediate frame wrapped or rewrote the message, the pair is
    /// discarded and the caller falls back to the untyped error. That makes a
    /// desync fail *safe* rather than fail *wrong*.
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
    /// Process-unique graph identity, assigned at construction and preserved
    /// across clones (a CoW working copy shares its parent's id; `version`
    /// distinguishes states). Never persisted — a loaded `.kgl` is a fresh
    /// runtime instance and gets a new id. Used with `version` as the Cypher
    /// plan-cache key so a cached plan can never leak across graphs.
    #[serde(skip, default = "next_graph_id")]
    pub graph_id: u64,
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
///
/// **Uniqueness is detective-by-design — preventive constraints are a
/// deliberate non-feature, not a gap.** Per-write `UNIQUE` / `NOT NULL` /
/// PRIMARY KEY constraints (à la Neo4j/Kùzu) are intentionally NOT
/// supported: they validate data-quality, which for an embedded
/// exploration/analytical engine belongs at load time (this batch O(n)
/// warning), not on the in-memory write hot path. The realistic needs are
/// already covered — `MERGE` (don't-duplicate upsert), this warning
/// (dirty-load signal), and the `db.duplicate_title` / `parallel_edges`
/// rule procedures (on-demand audit). Re-evaluate only for a concrete
/// interactive-write / untrusted-input workflow; even then, scope to
/// opt-in `id`-uniqueness (cheap — piggybacks `id_indices`), never an
/// arbitrary-property index maintained per write.
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

    /// Process-unique graph identity (see the `graph_id` field). Pairs with
    /// [`Self::version`] to form the Cypher plan-cache key.
    pub fn graph_id(&self) -> u64 {
        self.graph_id
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
    /// change. Monotonic; wraps only after 2^64 mutations (never in practice).
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
            unique_constraint_keys: Vec::new(),
            constraint_names: HashMap::new(),
            ddl_not_null_constraints: std::collections::BTreeSet::new(),
            id_indices: IdIndexStore::new(),
            connection_types: std::collections::HashSet::new(),
            node_type_metadata: HashMap::new(),
            connection_type_metadata: HashMap::new(),
            save_metadata: SaveMetadata::current(),
            id_field_aliases: FxHashMap::default(),
            title_field_aliases: FxHashMap::default(),
            parent_types: HashMap::new(),
            graph_instructions: HashMap::new(),
            user_schema_version: 0,
            checkpoint_lsn: 0,
            auto_vacuum_threshold: default_auto_vacuum_threshold(),
            spatial_configs: HashMap::new(),
            wkt_cache: Arc::new(RwLock::new(HashMap::new())),
            edge_type_counts_cache: Default::default(),
            type_connectivity_cache: Default::default(),
            property_ndv_cache: Arc::new(RwLock::new((0, HashMap::new()))),
            embeddings: HashMap::new(),
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
            type_schemas: HashMap::new(),
            has_secondary_labels: false,
            secondary_label_index: HashMap::new(),
        }
    }

    /// Create a DirGraph from a pre-existing graph (used by v3 loader).
    /// All metadata fields start empty and are populated by the caller.
    pub fn from_graph(graph: GraphBackend) -> Self {
        DirGraph {
            graph_id: next_graph_id(),
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
            unique_constraint_keys: Vec::new(),
            constraint_names: HashMap::new(),
            ddl_not_null_constraints: std::collections::BTreeSet::new(),
            id_indices: IdIndexStore::new(),
            connection_types: std::collections::HashSet::new(),
            node_type_metadata: HashMap::new(),
            connection_type_metadata: HashMap::new(),
            save_metadata: SaveMetadata::default(),
            id_field_aliases: FxHashMap::default(),
            title_field_aliases: FxHashMap::default(),
            parent_types: HashMap::new(),
            graph_instructions: HashMap::new(),
            user_schema_version: 0,
            checkpoint_lsn: 0,
            auto_vacuum_threshold: default_auto_vacuum_threshold(),
            spatial_configs: HashMap::new(),
            wkt_cache: Arc::new(RwLock::new(HashMap::new())),
            edge_type_counts_cache: Default::default(),
            type_connectivity_cache: Default::default(),
            property_ndv_cache: Arc::new(RwLock::new((0, HashMap::new()))),
            embeddings: HashMap::new(),
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
            None => return TypeIdIndex::General(HashMap::new()),
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

        // Fallback: scan all edges (pre-metadata graphs). On the disk
        // backend `edge_weights()` materializes into the query arena,
        // which must run under a DiskQueryGuard (arena protocol in
        // disk/graph.rs, enforced by a debug assert).
        let _guard = self.graph.begin_query();
        for edge in self.graph.edge_weights() {
            self.connection_types.insert(edge.connection_type);
        }
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

    pub fn get_node(&self, index: NodeIndex) -> Option<&NodeData> {
        self.graph.node_weight(index)
    }

    // ── Column stores: DirGraph is the access point, the backend is the owner ──
    //
    // D1 Phase 3 moved the per-type `ColumnStore` map onto the storage backend
    // (`MemoryGraph` / `MappedGraph` / `DiskGraph` all carry one now) and
    // deleted `DirGraph.column_stores` along with both halves of the
    // DirGraph↔DiskGraph mirror that used to keep two copies in step. DirGraph
    // keeps every lifecycle entry point — `enable_columnar`, `save`, spill,
    // vacuum — and reaches the stores through these delegates, which translate
    // the type *name* callers use into the `InternedKey` the backend keys by.

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

    /// Install (or replace) `node_type`'s store.
    #[inline]
    pub fn install_column_store(&mut self, node_type: &str, store: Arc<ColumnStore>) {
        self.graph
            .install_column_store(InternedKey::from_str(node_type), store);
    }

    /// Remove and return `node_type`'s store.
    #[inline]
    pub fn take_column_store(&mut self, node_type: &str) -> Option<Arc<ColumnStore>> {
        self.graph
            .take_column_store(InternedKey::from_str(node_type))
    }

    /// Drop every column store this graph owns.
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

    /// Number of types with a column store.
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
        // Declared UNIQUE constraints persist the same way. `unique_indices`
        // keys *are* the declaration list, so snapshotting them keeps the two
        // from drifting when a constraint is dropped.
        //
        // Sorted, unlike the index-key lists above: `unique_indices` is a
        // `HashMap`, so its iteration order is reseeded per process and two
        // saves of the same graph would otherwise disagree byte for byte. The
        // order carries no meaning — `rebuild_unique_indices_from_keys` reads
        // the list as a set — so imposing one costs nothing and makes a saved
        // graph reproducible.
        let mut unique_keys: Vec<UniqueConstraintKey> =
            self.unique_indices.keys().cloned().collect();
        unique_keys.sort_unstable();
        self.unique_constraint_keys = unique_keys;
        // Constraint *names* cannot be re-derived from the enforcement
        // structures, so unlike the lists above they are maintained live. Prune
        // instead: a name whose declaration is gone must not be saved, or
        // `DROP CONSTRAINT <name>` would resurrect it after a reload.
        self.prune_constraint_names();
    }

    /// Rebuild property and composite indexes from the persisted key lists.
    /// Called automatically after load.
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

        let _preexisting_violations = self.rebuild_unique_indices_from_keys();
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
        // Group on the node's *interned* type key, not its name. The name is
        // a per-type fact but this loop is per-node: resolving it and
        // allocating a `String` for every node — then hashing that string
        // with SipHash on the way into the map — was ~10% of a fired vacuum
        // at 1M. `InternedKey` is a `Copy` integer, so grouping on it costs
        // nothing and the O(types) resolve happens once, below.
        let mut by_type_key: FxHashMap<InternedKey, Vec<NodeIndex>> =
            FxHashMap::with_capacity_and_hasher(type_count, Default::default());
        {
            // Arena guard: node_weight materializes on the disk backend
            // (protocol in disk/graph.rs); scoped so the borrow ends before
            // the replace_with below.
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
        let mut new_type_indices: HashMap<String, Vec<NodeIndex>> =
            HashMap::with_capacity(by_type_key.len());
        for (type_key, indices) in by_type_key {
            new_type_indices.insert(self.interner.resolve(type_key).to_string(), indices);
        }
        self.type_indices.replace_with(new_type_indices);
        // `secondary_label_index` is *not* rebuilt from node data — it's
        // the canonical store, populated either by the choke-point API
        // during the session or by the load path (the disk sidecar /
        // the in-memory .kgl section).
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

        // Fallback: if metadata is empty (pre-metadata graph), scan nodes.
        // Arena guard: node_weight materializes on the disk backend
        // (protocol in disk/graph.rs); scoped so the borrow ends before
        // Phase 3's node_weight_mut.
        if schemas.is_empty() {
            let _guard = self.graph.begin_query();
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

        // Fallback: if metadata is empty (loaded from file), scan nodes.
        // Arena guard: node_weight materializes on the disk backend
        // (protocol in disk/graph.rs); scoped so the borrow ends before
        // the single-pass node_weight_mut loop below.
        if schemas.is_empty() {
            let _guard = self.graph.begin_query();
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

    /// Convert all node properties from Compact to Columnar storage.
    /// Properties are moved into per-type `ColumnStore` instances.
    /// This reduces memory usage by eliminating per-node `Value` enum overhead
    /// for homogeneous typed columns.
    ///
    /// Idempotent fast path: returns early when (a) every live node
    /// is already in `PropertyStorage::Columnar`, AND (b) every
    /// node's `Arc<ColumnStore>` is identical to the one in
    /// the backend's store for its type. Without this guard, a
    /// second `g.save()` after a successful first save runs the
    /// full `for node in graph` rebuild loop against already-
    /// Columnar properties — at wiki100m that's ~257 s
    /// (820 µs/node × 938 k nodes) — purely wasted work. Mapped
    /// graphs from `load_ntriples` are also already fully columnar
    /// (linked via `build_columns_direct`'s second-pass), so the
    /// same fast-path applies. 0.8.16.
    ///
    /// # Why the fast path is still sound without the Arc-identity check
    ///
    /// Before D1 Phase 3 this guard compared each node's own store `Arc` with
    /// the graph's by pointer, because `PropertyStorage::insert` on a columnar
    /// node did `Arc::make_mut` and forked the node away from the master —
    /// an `add_nodes(conflict_handling="update")` followed by `save()` would
    /// otherwise silently drop the new properties.
    ///
    /// That fork is now **inexpressible**: a node holds a row id, not a handle,
    /// and every columnar write goes through
    /// [`GraphWrite::set_node_property`](crate::graph::storage::GraphWrite::set_node_property),
    /// which mutates the one store the backend owns. There is no second replica
    /// to diverge, so the pointer comparison could only ever return "same" and
    /// has been deleted.
    ///
    /// The two checks that remain are the ones detecting state the store cannot
    /// see at all, and both are still required:
    ///
    /// - **inline-title divergence** — an in-place title write sets the node's
    ///   inline `title` field, not the store's `__title__` column;
    /// - **orphaned rows** — `DETACH DELETE` removes the node but leaves its
    ///   row, so `sum(row_count) != node_count`.
    ///
    /// Losing this fast path would cost a full O(N) rebuild on every save
    /// (~257 s at wiki100m), so it is pinned by
    /// `column_ownership_tests::a_second_save_of_an_unmodified_graph_skips_the_rebuild`,
    /// which counts rebuilds rather than trusting the reasoning above.
    pub fn enable_columnar(&mut self) {
        if self.column_store_count() > 0 && self.is_columnar() {
            // Arena guard: node_weight materializes on the disk backend
            // (protocol in disk/graph.rs); the whole drift check is
            // read-only and the guard drops at the end of this block.
            let _guard = self.graph.begin_query();
            let backend = &self.graph;
            let any_drift = self
                .graph
                .node_indices()
                .filter_map(|idx| self.graph.node_weight(idx))
                .any(|n| match &n.properties {
                    PropertyStorage::Columnar(row) => {
                        let row_id = row.row_id();
                        match backend.column_store(n.node_type) {
                            Some(graph_store) => {
                                // An in-place title write (Cypher `SET n.title`,
                                // add_nodes update/replace, connection titles)
                                // sets the inline `node.title` but not the
                                // columnar `__title__`. Detect that divergence
                                // so we rebuild and consolidate the fresh title
                                // (the title-only path doesn't clone the store,
                                // so it wouldn't otherwise register as drift —
                                // petekSuite bug 2). A consolidated/loaded node
                                // has `node.title == Null`, so no false drift.
                                !matches!(n.title, Value::Null)
                                    && graph_store.get_title(row_id).as_ref() != Some(&n.title)
                            }
                            None => true,
                        }
                    }
                    _ => true,
                });
            // Detect deletions that orphaned column rows. `DETACH DELETE`
            // removes the node from the topology but leaves the master column
            // store untouched, so total store rows exceed the live node count.
            // Without this the early-return below serialized the STALE store —
            // the deleted row (id/title/props) survived reload as a "ghost":
            // findable by id-lookup, re-bound by MERGE, and inconsistent with
            // the live count (petekSuite bug 5). Deletes only ever leave
            // store_rows >= live, so a total mismatch reliably flags them
            // (per-node adds/edits are already caught by `any_drift`). O(types),
            // so the clean fast-path stays cheap. Force a rebuild from live
            // nodes when they diverge.
            let total_store_rows: u64 = self
                .graph
                .column_stores_iter()
                .map(|(_, s)| s.row_count() as u64)
                .sum();
            let orphaned_rows = total_store_rows != self.graph.node_count() as u64;
            if !any_drift && !orphaned_rows {
                return;
            }
        }
        self.rebuild_column_stores();
    }

    /// The O(N) half of [`DirGraph::enable_columnar`]: rebuild every type's
    /// store from live nodes and re-point the nodes at their rows.
    ///
    /// Split out so the idempotence guard above reads as the decision it is,
    /// and so the rebuild has a single entry point to count (D1 risk 1).
    fn rebuild_column_stores(&mut self) {
        #[cfg(test)]
        note_columnar_rebuild();
        {}
        use crate::graph::storage::column_store::ColumnStore;

        // Ensure properties are compacted first
        if self.type_schemas.is_empty() {
            self.compact_properties();
        }

        // Build a ColumnStore per node type
        let mut stores: HashMap<String, ColumnStore> = HashMap::new();
        // Track row_id assignment per type
        let mut row_ids: HashMap<String, HashMap<NodeIndex, u32>> = HashMap::new();

        // Clean type_indices: remove entries for deleted/tombstoned nodes.
        // Arena guard: node_weight materializes on the disk backend
        // (protocol in disk/graph.rs); block-scoped read.
        {
            let _guard = self.graph.begin_query();
            let graph_ref = &self.graph;
            self.type_indices
                .retain_all(|idx| graph_ref.node_weight(*idx).is_some());
        }

        // First pass: create stores and push rows. Arena guard: node_weight
        // materializes on the disk backend (protocol in disk/graph.rs);
        // scoped so the borrow ends before the second pass's
        // node_weight_mut re-pointing below.
        let first_pass_guard = self.graph.begin_query();
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

            // Build column rows in ascending node-index order so the saved row
            // order matches the load-side re-point, which enumerates
            // `type_indices` rebuilt in ascending node-index order (see
            // io/file.rs "Re-point nodes to columnar storage" +
            // rebuild_type_indices_and_compact scanning node_indices()). This
            // `type_indices` may be in insertion order, which diverges from
            // index order once a node has been deleted (the free slot is reused
            // or a hole remains). Left unsorted, save wrote row k from the k-th
            // *inserted* node while load bound row k to the k-th *ascending*
            // node — rebinding every row's id/title/props to the wrong node and
            // scrambling edges on reload (petekSuite bug 4). Sorting here is the
            // single point that guarantees save-order == load-order.
            let mut sorted_indices: Vec<NodeIndex> = indices.iter().collect();
            sorted_indices.sort_unstable_by_key(|i| i.index());
            for idx in sorted_indices {
                if let Some(node) = self.graph.node_weight(idx) {
                    // Push id/title for every node. For Columnar nodes, read from
                    // the old column store. For Compact/Map nodes, use node.id/title.
                    // Always push id and title. For Columnar nodes, try old store first,
                    // fall back to node fields. For Compact/Map, use node fields directly.
                    let old_row = node.properties.columnar_row_id();
                    let old_store = old_row.and(self.graph.column_store(node.node_type));
                    let id_val = if let (Some(old_store), Some(old_row)) = (old_store, old_row) {
                        old_store.get_id(old_row).unwrap_or_else(|| node.id.clone())
                    } else {
                        node.id.clone()
                    };
                    let title_val = if let (Some(old_store), Some(old_row)) = (old_store, old_row) {
                        // Prefer a non-null inline `node.title` override. Every
                        // in-place title write (Cypher `SET n.title`, add_nodes
                        // update/replace, connection-title updates) sets the
                        // inline field but not necessarily the columnar
                        // `__title__`; reading only `old_store.get_title` here
                        // re-consolidated the STALE column value, so titles
                        // reverted on save+reload (petekSuite bug 2). A loaded,
                        // untouched columnar node has `node.title == Null`
                        // (nulled at load), so it correctly falls back to the
                        // store.
                        if !matches!(node.title, Value::Null) {
                            node.title.clone()
                        } else {
                            old_store
                                .get_title(old_row)
                                .unwrap_or_else(|| node.title.clone())
                        }
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
                        PropertyStorage::Columnar(row) => self
                            .graph
                            .column_store(node.node_type)
                            .map(|store| store.row_properties(row.row_id()))
                            .unwrap_or_default(),
                    };

                    let row_id = store.push_row(&pairs);
                    type_row_ids.insert(idx, row_id);
                }
            }

            stores.insert(node_type.to_string(), store);
            row_ids.insert(node_type.to_string(), type_row_ids);
        }
        drop(first_pass_guard);

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

        self.install_rebuilt_column_stores(stores, &row_ids);
    }

    /// Second pass of the rebuild: publish the stores onto the backend and
    /// point each node at its row.
    fn install_rebuilt_column_stores(
        &mut self,
        stores: HashMap<String, ColumnStore>,
        row_ids: &HashMap<String, HashMap<NodeIndex, u32>>,
    ) {
        let arc_stores: HashMap<String, Arc<ColumnStore>> =
            stores.into_iter().map(|(t, s)| (t, Arc::new(s))).collect();

        for (node_type, type_row_ids) in row_ids {
            if arc_stores.contains_key(node_type) {
                for (&idx, &row_id) in type_row_ids {
                    if let Some(node) = self.graph.node_weight_mut(idx) {
                        node.properties = PropertyStorage::Columnar(ColumnarRow::new(row_id));
                        // id/title were pushed into the store's reserved
                        // __id__/__title__ columns in the first pass, so the
                        // inline copies are now redundant. Null them to the
                        // sentinel (the backend reads them through the store)
                        // — otherwise topology serialization writes every
                        // id/title twice (inline + column section), bloating
                        // the saved file by ~27 B/node. Mirrors the load path
                        // (io/file/columns.rs) and the mapped batch path
                        // (mutation/batch.rs), which both null here.
                        node.id = Value::Null;
                        node.title = Value::Null;
                    }
                }
            }
        }

        // Install on the backend — the sole owner.
        self.clear_column_stores();
        for (node_type, store) in arc_stores {
            self.install_column_store(&node_type, store);
        }
    }

    /// Convert all Columnar properties back to Compact.
    /// Used when a caller needs a self-contained non-columnar graph.
    pub fn disable_columnar(&mut self) {
        // Per **type**, not per node. A node and its store are both owned by
        // the backend, so a `node_weight_mut` borrow cannot also read the
        // store — but cloning the type's store `Arc` once (O(1), a refcount
        // bump) releases the backend borrow for the whole inner loop. That
        // keeps this a single pass over nodes and hoists the two per-node
        // allocations D1 Phase 3 briefly introduced: the type name `String`
        // and the `TypeSchema` `Arc` clone are per-type facts, resolved once.
        let type_keys: Vec<InternedKey> = self
            .graph
            .column_stores_iter()
            .map(|(key, _)| key)
            .collect();

        for type_key in type_keys {
            let Some(type_str) = self.interner.try_resolve(type_key).map(str::to_string) else {
                continue;
            };
            let Some(store) = self.graph.column_store(type_key).map(Arc::clone) else {
                continue;
            };
            let schema = self.type_schemas.get(&type_str).cloned();
            let indices: Vec<NodeIndex> = self
                .type_indices
                .get(&type_str)
                .map(|set| set.iter().collect())
                .unwrap_or_default();

            for idx in indices {
                let Some(node) = self.graph.node_weight_mut(idx) else {
                    continue;
                };
                let Some(row_id) = node.properties.columnar_row_id() else {
                    continue;
                };
                // `row_properties` excludes the reserved `__id__`/`__title__`
                // columns, so a null-sentinel node (set by `enable_columnar` or
                // by the load path) would lose its identity when the columnar
                // link drops. Pull both back.
                if matches!(node.id, Value::Null) {
                    if let Some(v) = store.get_id(row_id) {
                        node.id = v;
                    }
                }
                if matches!(node.title, Value::Null) {
                    if let Some(v) = store.get_title(row_id) {
                        node.title = v;
                    }
                }
                let pairs = store.row_properties(row_id);
                node.properties = match &schema {
                    Some(schema) => PropertyStorage::from_compact(pairs, schema),
                    None => PropertyStorage::Map(pairs.into_iter().collect()),
                };
            }
        }
        self.clear_column_stores();
    }

    /// Returns true if any nodes are using columnar storage.
    pub fn is_columnar(&self) -> bool {
        self.graph.has_column_stores()
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

        let need_create = if let Some(existing) = self.column_store(node_type) {
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

            if let Some(old_arc) = self.take_column_store(node_type) {
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
                self.install_column_store(node_type, Arc::new(new_store));
            } else {
                let store = ColumnStore::new(current_schema, &meta, &self.interner);
                self.install_column_store(node_type, Arc::new(store));
            }
        }

        Arc::make_mut(self.column_store_mut(node_type).unwrap())
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
    /// The store it mutates is the backend's own (D1 Phase 3), so there is no
    /// read-side copy to push to afterwards — the pre-D1 shape kept a second
    /// map on `DirGraph` and needed an explicit sync per batch.
    /// Check heap usage of column stores and spill largest to disk if over limit.
    /// No-op if memory_limit is None or the backend is memory-mode.
    pub fn maybe_spill_columns(&mut self) {
        let limit = match self.memory_limit {
            Some(l) => l,
            None => return,
        };
        let total: usize = self
            .graph
            .column_stores_iter()
            .map(|(_, s)| s.heap_bytes())
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
        let mut by_size: Vec<(String, usize)> = self
            .column_stores_by_name()
            .into_iter()
            .map(|(t, s)| (t.to_string(), s.heap_bytes()))
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
            // heap it materialises to files (D1 defect 2). Before Phase 3 every
            // node held a strong `Arc`, so this forked and reclaimed nothing.
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
    /// update any external references (e.g., selections). An empty map means
    /// nothing was remapped.
    ///
    /// No-op if there are no tombstones (node_count == node_bound), and a
    /// no-op on the **disk** backend: its CSR arrays are frozen mmap, not a
    /// `StableDiGraph`, so there is no petgraph tombstone to compact — disk
    /// reclaims space by publishing a fresh generation (`compact_disk`), not
    /// by rebuilding in place. Rebuilding would also have to materialise the
    /// whole graph on the heap, which is the one thing the disk backend
    /// exists to avoid.
    ///
    /// The rebuild preserves the backend variant and any write-capture
    /// wrapper. **Callers on a durable graph must flush the write-ahead log
    /// first**: buffered ops are keyed by `NodeIndex` and every index moves
    /// here.
    pub fn vacuum(&mut self) -> HashMap<NodeIndex, NodeIndex> {
        if self.graph.is_disk() {
            return HashMap::new();
        }
        let old_node_count = self.graph.node_count();
        let old_node_bound = self.graph.node_bound();

        // No petgraph tombstones — but columnar stores may still have orphaned rows
        // (e.g., all nodes deleted → petgraph is empty but column data remains).
        if old_node_count == old_node_bound {
            let columnar_orphaned = self.column_stores_by_name().into_iter().any(|(t, s)| {
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

        // Take ownership of the old graph so the rebuild can *relocate* every
        // weight instead of deep-cloning it. A `NodeData` clone reallocates
        // the id/title strings and the whole property vector, and the
        // originals were dropped moments later when the backend was replaced
        // — the clone/drop pair was pure waste (profiled at ~16% of a fired
        // vacuum at 1M, plus its share of allocator and memcpy time).
        let Some(mut old) = self.graph.take_heap_graph() else {
            // Disk: nothing was taken, so nothing downstream may treat the
            // indices as remapped.
            return HashMap::new();
        };

        // Build new graph with contiguous indices
        let mut new_graph = StableDiGraph::with_capacity(old_node_count, old.edge_count());
        let mut old_to_new: HashMap<NodeIndex, NodeIndex> = HashMap::with_capacity(old_node_count);
        // Dense old→new lookup for the edge pass. Endpoint remapping is two
        // probes per edge, and running them through the returned map's
        // SipHash was the largest single cost in the rebuild (hashing was
        // ~22% of a fired vacuum at 1M). The graph is index-addressed, so a
        // bound-sized vector is the natural map; `u32::MAX` marks a slot that
        // held no live node.
        let mut dense: Vec<u32> = vec![u32::MAX; old_node_bound];

        // Move all live nodes over, recording the index mapping. Ascending
        // raw order reproduces `node_indices()` exactly, so the compacted
        // indices are the same ones the clone loop produced.
        for (raw, mapped) in dense.iter_mut().enumerate() {
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
            *mapped = new_idx.index() as u32;
            old_to_new.insert(old_idx, new_idx);
        }

        // Move all live edges over with remapped endpoints. The ids are
        // collected because relocating a weight needs `&mut old` while
        // `edge_indices()` borrows it.
        let old_edge_ids: Vec<EdgeIndex> = old.edge_indices().collect();
        for old_edge_idx in old_edge_ids {
            let Some((src, tgt)) = old.edge_endpoints(old_edge_idx) else {
                continue;
            };
            let (new_src, new_tgt) = (dense[src.index()], dense[tgt.index()]);
            debug_assert!(
                new_src != u32::MAX && new_tgt != u32::MAX,
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
            return HashMap::new();
        }

        // Remap embedding stores to use new node indices (see embedding_carry.rs).
        self.remap_embedding_slots(&old_to_new);

        // Rebuild all indexes from the compacted graph
        self.reindex();

        // Rebuild columnar stores if active — old stores have orphaned rows
        // from deleted nodes. The disable/enable cycle reads only live nodes,
        // producing fresh ColumnStores with no dead rows.
        if self.is_columnar() {
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
                .graph
                .column_stores_iter()
                .map(|(_, s)| s.row_count() as usize)
                .sum(),
            columnar_live_rows: self
                .column_stores_by_name()
                .into_iter()
                .map(|(t, _)| self.type_indices.get(t).map(|v| v.len()).unwrap_or(0))
                .sum(),
            columnar_heap_bytes: self
                .graph
                .column_stores_iter()
                .map(|(_, s)| s.heap_bytes())
                .sum(),
            columnar_is_mapped: self.graph.column_stores_iter().any(|(_, s)| s.is_mapped()),
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
    /// Heap bytes held by the column stores the backend owns.
    ///
    /// Lifted into `GraphInfo` by D1 Phase 3 so a binding can report columnar
    /// memory without reaching into storage: the stores are backend-owned and
    /// there is no `DirGraph.column_stores` field to read any more.
    pub columnar_heap_bytes: usize,
    /// `true` when at least one column store has been spilled to mmap.
    pub columnar_is_mapped: bool,
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
