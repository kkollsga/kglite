use crate::datatypes::Value;
use crate::graph::schema::{
    DirGraph, EdgeData, InternedKey, NodeData, PropertyStorage, PROVISIONAL_KEY,
};
use crate::graph::storage::column_store::ColumnStore;
use crate::graph::storage::property_storage::ColumnarRow;
use crate::graph::storage::{GraphRead, GraphWrite};
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::Direction;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

/// `(node, row id)` for every node whose columnar row reference has to be
/// attached once the append finishes.
type DeferredColumnarRows = Vec<(NodeIndex, u32)>;

/// Column stores held outright (refcount 1) for the duration of an append,
/// keyed by node type — see `BatchProcessor::detach_columnar_stores`.
type OwnedColumnStores = HashMap<String, ColumnStore>;

const SMALL_BATCH_THRESHOLD: usize = 100;
const MEDIUM_BATCH_THRESHOLD: usize = 1000;
const LARGE_BATCH_CHUNK_SIZE: usize = 1000;

#[derive(Debug)]
enum BatchType {
    Small,
    Medium,
    Large,
}

#[derive(Debug, Default)]
pub struct BatchMetrics {
    pub processing_time: f64,
    pub memory_used: usize,
    pub batch_count: usize,
}

#[derive(Debug)]
pub enum NodeAction {
    Update {
        node_idx: NodeIndex,
        title: Option<Value>, // None leaves the existing title untouched
        /// Pre-interned like `CreateInterned`: resolving these back to `String`
        /// only to re-intern them in `apply_updates` cost an allocation, a
        /// `SipHash` map insert and an interner probe per property per row.
        properties: Vec<(InternedKey, Value)>,
        conflict_mode: ConflictHandling,
    },
    /// Create with pre-interned property keys (avoids re-interning per row)
    CreateInterned {
        node_type: String,
        id: Value,
        title: Value,
        properties: Vec<(InternedKey, Value)>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ConflictHandling {
    Replace, // Replace all properties and title (whole-node overwrite)
    Skip,    // Don't update existing nodes/edges
    #[default]
    // Merge: write the incoming properties, leave properties NOT in this batch
    // untouched. STABLE CONTRACT (partial-update guarantee): a reload can
    // re-assert a subset of fields without clobbering fields another writer
    // owns — see the regression test `add_nodes_update_is_partial` and the
    // `add_nodes` pyi docstring. Drives the managed-reload guard.
    Update,
    Preserve, // Merge properties, existing values take precedence
    Sum,      // Merge properties, add numeric values (edges); acts as Update for nodes
}

/// Add two Values if both are numeric. Mixed Int64+Float64 promotes to Float64.
/// Non-numeric values fall back to Update behavior (new value overwrites).
pub(crate) fn sum_values(existing: &Value, new: &Value) -> Value {
    match (existing, new) {
        (Value::Int64(a), Value::Int64(b)) => Value::Int64(a.wrapping_add(*b)),
        (Value::Float64(a), Value::Float64(b)) => Value::Float64(a + b),
        (Value::Int64(a), Value::Float64(b)) => Value::Float64(*a as f64 + b),
        (Value::Float64(a), Value::Int64(b)) => Value::Float64(a + *b as f64),
        _ => new.clone(),
    }
}

#[derive(Debug)]
struct NodeCreationInterned {
    node_type: String,
    id: Value,
    title: Value,
    properties: Vec<(InternedKey, Value)>,
}

#[derive(Debug)]
struct NodeUpdate {
    node_idx: NodeIndex,
    title: Option<Value>,
    properties: Vec<(InternedKey, Value)>,
    conflict_mode: ConflictHandling,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct BatchStats {
    pub creates: usize,
    pub updates: usize,
}

impl BatchStats {
    fn combine(&mut self, other: &BatchStats) {
        self.creates += other.creates;
        self.updates += other.updates;
    }
}

#[derive(Debug)]
pub struct BatchProcessor {
    creates_interned: Vec<NodeCreationInterned>,
    updates: Vec<NodeUpdate>,
    capacity: usize,
    batch_type: BatchType,
    metrics: BatchMetrics,
    accumulated_stats: BatchStats,
}

impl BatchProcessor {
    pub fn new(estimated_size: usize) -> Self {
        let (capacity, batch_type) = match estimated_size {
            n if n < SMALL_BATCH_THRESHOLD => (n, BatchType::Small),
            n if n < MEDIUM_BATCH_THRESHOLD => (n, BatchType::Medium),
            _ => (LARGE_BATCH_CHUNK_SIZE, BatchType::Large),
        };

        BatchProcessor {
            creates_interned: Vec::with_capacity(capacity),
            updates: Vec::with_capacity(capacity),
            capacity,
            batch_type,
            metrics: BatchMetrics::default(),
            accumulated_stats: BatchStats::default(),
        }
    }

    pub fn add_action(&mut self, action: NodeAction, graph: &mut DirGraph) -> Result<(), String> {
        match action {
            NodeAction::CreateInterned {
                node_type,
                id,
                title,
                properties,
            } => {
                self.creates_interned.push(NodeCreationInterned {
                    node_type,
                    id,
                    title,
                    properties,
                });
            }
            NodeAction::Update {
                node_idx,
                title,
                properties,
                conflict_mode,
            } => {
                self.updates.push(NodeUpdate {
                    node_idx,
                    title,
                    properties,
                    conflict_mode,
                });
            }
        }

        if let BatchType::Large = self.batch_type {
            if self.creates_interned.len() >= self.capacity {
                let stats = self.flush_chunk(graph)?;
                self.accumulated_stats.combine(&stats);
            }
        }

        Ok(())
    }

    fn flush_chunk(&mut self, graph: &mut DirGraph) -> Result<BatchStats, String> {
        let start = Instant::now();
        let mut stats = BatchStats::default();
        // Two passes on every backend, to keep a bulk append linear: pass 1
        // owns each affected type's store outright while rows are pushed into
        // it ([`Self::detach_columnar_stores`]), pass 2 installs the stores
        // again and points every created node at its row
        // ([`Self::reattach_columnar_stores`]). The pair used to run for
        // mapped/disk only, with memory building row-shaped nodes instead;
        // construction is columnar in every mode now (see
        // `dir_graph::node_write`), so there is one path.
        let mut deferred_columnar: DeferredColumnarRows = Vec::new();
        let mut owned_stores: OwnedColumnStores =
            Self::detach_columnar_stores(&self.creates_interned, graph);

        // The store this chunk is currently appending to, held out of the map
        // for as long as consecutive rows share a type — which is every row of
        // a `add_nodes` DataFrame. Looking it up in `owned_stores` per row cost
        // two `String`-keyed (SipHash) probes on a path that then does one
        // column push; a single `String` comparison replaces both.
        let mut current: Option<(String, ColumnStore)> = None;

        for creation in self.creates_interned.drain(..) {
            let type_key = graph.interner.get_or_intern(&creation.node_type);

            // Pass 1: push into the owned ColumnStore, straight from the
            // interned pairs the caller handed us — folding them into a
            // `NodeData` `HashMap` first and draining it back out cost a map
            // allocation and a round trip per node created.
            let row_id = {
                if current
                    .as_ref()
                    .is_none_or(|(held, _)| *held != creation.node_type)
                {
                    if let Some((held, store)) = current.take() {
                        owned_stores.insert(held, store);
                    }
                    let store = owned_stores.remove(&creation.node_type).unwrap_or_else(|| {
                        let schema = graph
                            .type_schemas
                            .get(&creation.node_type)
                            .cloned()
                            .unwrap_or_else(|| Arc::new(crate::graph::schema::TypeSchema::new()));
                        let meta = graph
                            .node_type_metadata
                            .get(&creation.node_type)
                            .cloned()
                            .unwrap_or_default();
                        ColumnStore::new(schema, &meta, &graph.interner)
                    });
                    current = Some((creation.node_type.clone(), store));
                }
                let store = &mut current.as_mut().expect("installed just above").1;
                // A key this store's schema has never seen appends one column
                // inside `push_row`, back-filled with nulls. This used to
                // rebuild the whole store instead — every row already in the
                // chunk re-pushed on every newly-seen key, which is quadratic
                // over a widening ingest stream. See
                // `DirGraph::ensure_column_store_for_push`, which carried the
                // same rebuild for the Cypher create path.
                store.push_id(&creation.id);
                store.push_title(&creation.title);
                store.push_row(&creation.properties)
            };

            // id/title live in the store's reserved columns; the inline fields
            // carry the `Null` sentinel every columnar producer leaves.
            let node_data = NodeData {
                id: Value::Null,
                title: Value::Null,
                node_type: type_key,
                properties: PropertyStorage::Columnar(ColumnarRow::new(row_id)),
            };
            let node_idx = GraphWrite::add_node(&mut graph.graph, node_data);

            deferred_columnar.push((node_idx, row_id));

            // Statement-rollback capture for the `type_indices` append — the
            // one above-storage index edit the storage seam does not already
            // cover. Dead weight today, deliberately: no Cypher path reaches
            // this funnel (see [`Self::detach_columnar_stores`]), so the cost
            // is one `Option` check per node.
            let bucket_was_new = !graph.type_indices.contains_key(&creation.node_type);
            graph
                .type_indices
                .push_to_type(&creation.node_type, node_idx);
            if let Some(journal) = graph.graph.undo_journal_mut() {
                journal.note_bucket_appended(
                    crate::graph::storage::undo::BucketId::NodeType(creation.node_type),
                    node_idx,
                    bucket_was_new,
                );
            }
            // id_indices is intentionally NOT updated incrementally here.
            // Writing into entry_or_default before any prior `build_id_index`
            // call would create a partial entry that subsequent lookups
            // would trust as complete (build_id_index short-circuits when
            // an entry exists). Leave id_indices alone: `maintain::add_nodes`
            // settles the type's entry at return time, folding in exactly the
            // members appended here (`fold_appended_ids_into_index`, which
            // reads this bucket's tail) or invalidating for a rebuild when it
            // cannot. A future creating caller that does neither would leave a
            // stale entry, which is why the fold is on the *caller's* side of
            // this boundary and not here.
            stats.creates += 1;
        }
        if let Some((held, store)) = current.take() {
            owned_stores.insert(held, store);
        }

        Self::reattach_columnar_stores(graph, deferred_columnar, owned_stores);

        self.apply_updates(graph, &mut stats);

        self.metrics.processing_time += start.elapsed().as_secs_f64();
        self.metrics.batch_count += 1;
        self.metrics.memory_used = self.creates_interned.capacity() + self.updates.capacity();

        Ok(stats)
    }

    /// Pass 1 of the columnar append: take each affected type's `ColumnStore`
    /// out of the graph and own it outright for the duration of the append.
    ///
    /// Taking it out is what keeps a bulk append linear. While the graph's map
    /// still holds the `Arc`, the first `Arc::make_mut` in the append loop
    /// clones the whole store — once per row, O(n²) overall. With that ref
    /// removed the refcount is 1, `try_unwrap` succeeds, and the append mutates
    /// in place. [`Self::reattach_columnar_stores`] installs the stores again.
    ///
    /// The store's *contents* are not covered by cell entries: this funnel
    /// swaps stores wholesale rather than writing cells, and `column_stores`
    /// lives on the storage backend, which `rollback::swap_data_scale` parks so
    /// the schema shell cannot restore it. What it journals instead, when a
    /// checkpoint is open, is the append pre-image per affected type — the swap
    /// only ever *appends* rows to the store it took out, so truncating back to
    /// the captured length undoes the whole chunk. Defensive today: no Cypher
    /// statement reaches this funnel (the executor creates nodes through
    /// `DirGraph::insert_node_routed`), but it is what would make a bulk-ingest
    /// `CALL` routed through here reversible instead of silently surviving a
    /// rollback.
    fn detach_columnar_stores(
        creates: &[NodeCreationInterned],
        graph: &mut DirGraph,
    ) -> OwnedColumnStores {
        let mut owned_stores: OwnedColumnStores = HashMap::new();
        let affected_types: HashSet<String> = creates.iter().map(|c| c.node_type.clone()).collect();
        for node_type in &affected_types {
            let store_was_new = graph.column_store(node_type).is_none();
            // Nodes hold a row id and no store handle, so the backend map is
            // the sole owner and `try_unwrap` succeeds outright. Existing row
            // ids stay valid across `materialize_for_append`.
            if let Some(arc_store) = graph.take_column_store(node_type) {
                let mut store = Arc::try_unwrap(arc_store).unwrap_or_else(|a| (*a).clone());
                let meta = graph
                    .node_type_metadata
                    .get(node_type)
                    .cloned()
                    .unwrap_or_default();
                store.materialize_for_append(&meta, &graph.interner);
                // After the materialization, never across it: it re-derives the
                // store's columns, so a pre-image taken on the far side names a
                // schema that no longer describes them.
                Self::journal_append_pre_image(graph, node_type, &store, store_was_new);
                owned_stores.insert(node_type.clone(), store);
            } else {
                // A type whose store this chunk creates: the undo is to drop it.
                Self::journal_append_pre_image(
                    graph,
                    node_type,
                    &crate::graph::storage::column_store::ColumnStore::new(
                        Arc::new(crate::graph::schema::TypeSchema::new()),
                        &HashMap::new(),
                        &graph.interner,
                    ),
                    store_was_new,
                );
            }
        }
        owned_stores
    }

    /// Journal the pre-image that reverses this chunk's appends to one type's
    /// store, when a statement checkpoint is open. No-op otherwise, which is
    /// every call today — see [`Self::detach_columnar_stores`].
    fn journal_append_pre_image(
        graph: &mut DirGraph,
        node_type: &str,
        store: &crate::graph::storage::column_store::ColumnStore,
        store_was_new: bool,
    ) {
        if graph.graph.undo_journal_mut().is_none() {
            return;
        }
        let type_key = graph.interner.get_or_intern(node_type);
        let captured = crate::graph::storage::undo::ColumnarAppendPreImage::capture(store);
        if let Some(journal) = graph.graph.undo_journal_mut() {
            captured.record(journal, type_key, store_was_new);
        }
    }

    /// Pass 2 of the columnar append: publish the owned stores back into the
    /// graph and point every row this chunk created at its store row.
    ///
    /// Those rows are already covered by the `RawOp::UpsertNode` that
    /// `add_node` pushed, which resolves against post-reattachment state.
    fn reattach_columnar_stores(
        graph: &mut DirGraph,
        deferred_columnar: DeferredColumnarRows,
        owned_stores: OwnedColumnStores,
    ) {
        // Install unconditionally: the stores must go back even when the batch
        // created no rows for an affected type, because pass 1 took them out.
        for (node_type, store) in owned_stores {
            graph.install_column_store(&node_type, Arc::new(store));
        }
        // Disk only, in effect: `node_weight_mut()` materializes a `NodeData`
        // into an arena that the next call clears, so the `Columnar` row
        // reference the create loop put on the node does not reach the disk
        // slot — `update_row_id` is what persists it. A no-op on the heap
        // backends, where the node itself already carries the row.
        for (node_idx, row_id) in deferred_columnar {
            GraphWrite::update_row_id(&mut graph.graph, node_idx, row_id);
        }
        // No disk-side sync: `install_column_store` above wrote into the
        // backend's own map, which is what disk reads resolve through.
    }

    /// Apply this chunk's pending updates: resolve each target's storage
    /// representation and dispatch to the matching writer.
    fn apply_updates(&mut self, graph: &mut DirGraph, stats: &mut BatchStats) {
        // Disk writes reach the backend's `ColumnStore` directly, memory and
        // mapped go through the node — O(types) `Arc::make_mut` clones per
        // chunk rather than O(rows). Why the split: [`Self::apply_row_update`].
        let is_disk = GraphRead::is_disk(&graph.graph);
        let mut disk_updates_applied = false;
        // Interned once outside the loop; only the Update/Sum arms use it.
        let provisional_key = graph.interner.get_or_intern(PROVISIONAL_KEY);

        for update in self.updates.drain(..) {
            if update.conflict_mode == ConflictHandling::Skip {
                continue;
            }

            let interned_props = update.properties;

            if is_disk {
                let (type_name, row_id) = match &graph.graph {
                    crate::graph::schema::GraphBackend::Disk(ref dg) => {
                        let slot = dg.node_slot(update.node_idx.index());
                        if !slot.is_alive() {
                            continue;
                        }
                        let type_key = InternedKey::from_u64(slot.node_type);
                        let type_name = graph.interner.resolve(type_key).to_string();
                        (type_name, slot.row_id)
                    }
                    _ => unreachable!("is_disk guard"),
                };

                let Some(arc_store) = graph.column_store_mut(&type_name) else {
                    continue;
                };
                let store = Arc::make_mut(arc_store);
                Self::apply_row_update(
                    store,
                    row_id,
                    update.title,
                    interned_props,
                    update.conflict_mode,
                    provisional_key,
                );
                disk_updates_applied = true;
                stats.updates += 1;
            } else if graph.graph.node_weight(update.node_idx).is_some() {
                Self::apply_node_update(
                    graph,
                    update.node_idx,
                    update.title,
                    interned_props,
                    update.conflict_mode,
                    provisional_key,
                );
                stats.updates += 1;
            }
        }

        // No re-sync: `column_store_mut` above mutated the backend's own store.
        let _ = disk_updates_applied;
    }

    /// Write one pending update into a columnar row — the disk
    /// representation, where the `ColumnStore` row *is* the node's property
    /// storage. Arm for arm the twin of [`Self::apply_node_update`]; the two
    /// stay separate because `node_weight_mut` on a disk graph materialises
    /// `NodeData` into an arena that the next `&mut` call drops, so a disk
    /// update has to reach the store directly.
    fn apply_row_update(
        store: &mut crate::graph::storage::column_store::ColumnStore,
        row_id: u32,
        title: Option<Value>,
        properties: Vec<(InternedKey, Value)>,
        conflict_mode: ConflictHandling,
        provisional_key: InternedKey,
    ) {
        match conflict_mode {
            ConflictHandling::Skip => unreachable!(),
            ConflictHandling::Replace => {
                if let Some(new_title) = title {
                    store.set_title(row_id, &new_title);
                }
                // Null out every currently-set property on this row
                // before applying the new set — matches heap
                // `PropertyStorage::replace_all` semantics.
                let existing: Vec<InternedKey> = store
                    .row_properties(row_id)
                    .into_iter()
                    .map(|(k, _)| k)
                    .collect();
                for k in existing {
                    store.set(row_id, k, &Value::Null, None);
                }
                for (k, v) in properties {
                    store.set(row_id, k, &v, None);
                }
            }
            ConflictHandling::Update | ConflictHandling::Sum => {
                if let Some(new_title) = title {
                    store.set_title(row_id, &new_title);
                }
                for (k, v) in properties {
                    store.set(row_id, k, &v, None);
                }
                // Promote: a real-row upsert clears the stub marker.
                if store.get(row_id, provisional_key).is_some() {
                    store.set(row_id, provisional_key, &Value::Null, None);
                }
            }
            ConflictHandling::Preserve => {
                if let Some(new_title) = title {
                    let cur = store.get_title(row_id).unwrap_or(Value::Null);
                    if matches!(cur, Value::Null) {
                        store.set_title(row_id, &new_title);
                    }
                }
                for (k, v) in properties {
                    if store.get(row_id, k).is_none() {
                        store.set(row_id, k, &v, None);
                    }
                }
            }
        }
    }

    /// Write one pending update through the backend — the memory/mapped twin
    /// of [`Self::apply_row_update`].
    ///
    /// Routed through `GraphWrite` rather than `&mut NodeData` because a
    /// columnar node's properties live in the store the backend owns; there is
    /// no per-node storage left to write into. Title stays an
    /// inline `NodeData` field and keeps its own short borrow.
    fn apply_node_update(
        graph: &mut DirGraph,
        node_idx: NodeIndex,
        title: Option<Value>,
        properties: Vec<(InternedKey, Value)>,
        conflict_mode: ConflictHandling,
        provisional_key: InternedKey,
    ) {
        let set_title = |graph: &mut DirGraph, value: Value| {
            GraphWrite::set_node_title(&mut graph.graph, node_idx, value);
        };
        match conflict_mode {
            ConflictHandling::Skip => unreachable!(),
            ConflictHandling::Replace => {
                if let Some(new_title) = title {
                    set_title(graph, new_title);
                }
                GraphWrite::replace_node_properties(&mut graph.graph, node_idx, properties);
            }
            ConflictHandling::Update | ConflictHandling::Sum => {
                if let Some(new_title) = title {
                    set_title(graph, new_title);
                }
                for (k, v) in properties {
                    GraphWrite::set_node_property(&mut graph.graph, node_idx, k, v);
                }
                // Promote: a real-row upsert clears the stub marker.
                if graph.graph.node_has_property(node_idx, provisional_key) {
                    GraphWrite::set_node_property(
                        &mut graph.graph,
                        node_idx,
                        provisional_key,
                        Value::Null,
                    );
                }
            }
            ConflictHandling::Preserve => {
                if let Some(new_title) = title {
                    if graph
                        .graph
                        .get_node_title(node_idx)
                        .is_none_or(|t| matches!(t, Value::Null))
                    {
                        set_title(graph, new_title);
                    }
                }
                for (k, v) in properties {
                    GraphWrite::set_node_property_if_absent(&mut graph.graph, node_idx, k, v);
                }
            }
        }
    }

    pub fn execute(mut self, graph: &mut DirGraph) -> Result<(BatchStats, BatchMetrics), String> {
        let mut total_stats = self.accumulated_stats;

        match self.batch_type {
            BatchType::Small | BatchType::Medium => {
                let stats = self.flush_chunk(graph)?;
                total_stats.combine(&stats);
            }
            BatchType::Large => {
                if !self.creates_interned.is_empty() || !self.updates.is_empty() {
                    let stats = self.flush_chunk(graph)?;
                    total_stats.combine(&stats);
                }
            }
        }

        // Honour the memory limit once the whole batch has landed — not per
        // chunk, which would re-spill a store the next chunk is about to append
        // to. This is what makes an in-process mapped graph actually mapped:
        // `StorageMode::Mapped` is `memory_limit = Some(0)`, and with the limit
        // enforced nowhere on the ingest path a mapped graph built by
        // `add_nodes` stayed wholly on the heap until its first write. A no-op
        // when no limit is set, which is the default-mode path.
        graph.maybe_spill_columns();

        Ok((total_stats, self.metrics))
    }
}

// Properties arrive already interned. `EdgeData` stores
// `Vec<(InternedKey, Value)>`, so a `HashMap<String, Value>` here would only
// have been re-hashed and re-interned per row on the way out — the callers
// (`add_connections` above all) resolve their column names to keys once per
// call instead.
#[derive(Debug)]
struct ConnectionCreation {
    source_idx: NodeIndex,
    target_idx: NodeIndex,
    properties: Vec<(InternedKey, Value)>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ConnectionBatchStats {
    pub connections_created: usize,
    pub properties_tracked: usize,
}

impl ConnectionBatchStats {
    fn combine(&mut self, other: &ConnectionBatchStats) {
        self.connections_created += other.connections_created;
        self.properties_tracked = self.properties_tracked.max(other.properties_tracked);
    }
}

#[derive(Debug)]
pub struct ConnectionBatchProcessor {
    connections: Vec<ConnectionCreation>,
    schema_properties: HashSet<InternedKey>,
    /// First observed concrete type per property key (Value::type_name
    /// vocabulary). Pre-fix the bulk loaders registered every edge property
    /// as "Unknown" — the schema procedures then reported untyped edge
    /// properties to every client (measured 2026-08-15: all 59 sodir rel
    /// properties showed `unknown` in G.V()'s Data Explorer), while the
    /// Cypher CREATE path recorded real types.
    schema_property_types: HashMap<InternedKey, &'static str>,
    capacity: usize,
    batch_type: BatchType,
    metrics: BatchMetrics,
    conflict_mode: ConflictHandling,
    accumulated_stats: ConnectionBatchStats,
    skip_existence_check: bool,
}

impl ConnectionBatchProcessor {
    pub fn new(estimated_size: usize) -> Self {
        let (capacity, batch_type) = match estimated_size {
            n if n < SMALL_BATCH_THRESHOLD => (n, BatchType::Small),
            n if n < MEDIUM_BATCH_THRESHOLD => (n, BatchType::Medium),
            _ => (LARGE_BATCH_CHUNK_SIZE, BatchType::Large),
        };

        ConnectionBatchProcessor {
            connections: Vec::with_capacity(capacity),
            schema_properties: HashSet::new(),
            schema_property_types: HashMap::new(),
            capacity,
            batch_type,
            metrics: BatchMetrics::default(),
            conflict_mode: ConflictHandling::Update,
            accumulated_stats: ConnectionBatchStats::default(),
            skip_existence_check: false,
        }
    }

    pub fn set_conflict_mode(&mut self, mode: ConflictHandling) {
        self.conflict_mode = mode;
    }

    /// Skip edge existence checks (safe when this connection type has no existing edges)
    pub fn set_skip_existence_check(&mut self, skip: bool) {
        self.skip_existence_check = skip;
    }

    pub fn add_connection(
        &mut self,
        source_idx: NodeIndex,
        target_idx: NodeIndex,
        mut properties: Vec<(InternedKey, Value)>,
        graph: &mut DirGraph,
        connection_type: &str,
    ) -> Result<(), String> {
        // Freshness provenance: stamp `updated_at` when this edge type opted in
        // (single chokepoint for every `add_connections` route; registered into
        // `schema_properties` below so the columnar edge store gets a slot).
        graph.inject_edge_provenance_interned(connection_type, &mut properties);
        if !self.skip_existence_check {
            let conn_type_key = graph.interner.get_or_intern(connection_type);
            let existing_edge = graph
                .graph
                .edges_connecting(source_idx, target_idx)
                .find(|e| e.weight().connection_type == conn_type_key)
                .map(|e| e.id());

            if existing_edge.is_some() && self.conflict_mode == ConflictHandling::Skip {
                return Ok(());
            }
        }

        // Registration follows the row, so a caller that skips its null cells
        // never sees an all-null column materialize in the connection type's
        // property list. A key that does arrive Null still registers, but
        // contributes no type — it stays "Unknown" until a concrete value
        // shows up (see [`Self::schema_property_types`]).
        for (key, value) in &properties {
            self.schema_properties.insert(*key);
            let type_name = value.type_name();
            if type_name != "Null" {
                self.schema_property_types.entry(*key).or_insert(type_name);
            }
        }

        self.connections.push(ConnectionCreation {
            source_idx,
            target_idx,
            properties,
        });

        if let BatchType::Large = self.batch_type {
            if self.connections.len() >= self.capacity {
                let stats = self.flush_chunk(graph, connection_type)?;
                self.accumulated_stats.combine(&stats);
            }
        }

        Ok(())
    }

    fn flush_chunk(
        &mut self,
        graph: &mut DirGraph,
        connection_type: &str,
    ) -> Result<ConnectionBatchStats, String> {
        let start = Instant::now();
        let mut stats = ConnectionBatchStats::default();

        let conn_type_key = graph.interner.get_or_intern(connection_type);

        // Per-flush (source, target) -> edge_id map over the chunk's unique
        // source set and this connection type. The per-edge
        // `edges_connecting().find()` walk it replaces was O(N * max_degree)
        // for hub-source fan-out into an *existing* connection type; this is
        // O(sum_of_unique_source_degrees). Mutated as we go, preserving that
        // code's within-chunk dedup semantics: two chunk entries with the same
        // (src, tgt) consolidate onto one edge.
        //
        // `skip_existence_check` (initial-load fast path) skips both the build
        // and the per-edge lookup — there are no existing edges of this type,
        // and within-chunk consolidation is the caller's job in that mode.
        let mut existing_lookup: HashMap<(NodeIndex, NodeIndex), EdgeIndex> = HashMap::new();
        if !self.skip_existence_check {
            let unique_sources: HashSet<NodeIndex> =
                self.connections.iter().map(|c| c.source_idx).collect();
            for src in &unique_sources {
                for edge_ref in graph.graph.edges_directed(*src, Direction::Outgoing) {
                    if edge_ref.weight().connection_type == conn_type_key {
                        existing_lookup.insert((*src, edge_ref.target()), edge_ref.id());
                    }
                }
            }
        }

        for conn in self.connections.drain(..) {
            let existing_edge = if self.skip_existence_check {
                None
            } else {
                existing_lookup
                    .get(&(conn.source_idx, conn.target_idx))
                    .copied()
            };

            if let Some(edge_idx) = existing_edge {
                match self.conflict_mode {
                    ConflictHandling::Skip => {
                        // Defensive: `add_connection` already filtered these.
                        continue;
                    }
                    ConflictHandling::Replace => {
                        GraphWrite::remove_edge(&mut graph.graph, edge_idx);
                        let edge_data = EdgeData::new_interned(conn_type_key, conn.properties);
                        let new_id = GraphWrite::add_edge(
                            &mut graph.graph,
                            conn.source_idx,
                            conn.target_idx,
                            edge_data,
                        );
                        // Update the lookup so any later chunk entry with
                        // the same (src, tgt) hits the freshly-created edge,
                        // not the removed one.
                        existing_lookup.insert((conn.source_idx, conn.target_idx), new_id);
                        stats.connections_created += 1;
                    }
                    ConflictHandling::Update => {
                        let interned_props = conn.properties;
                        if let Some(EdgeData {
                            properties: edge_props,
                            ..
                        }) = GraphWrite::edge_weight_mut(&mut graph.graph, edge_idx)
                        {
                            for (k, v) in interned_props {
                                if let Some((_, existing)) =
                                    edge_props.iter_mut().find(|(ek, _)| *ek == k)
                                {
                                    *existing = v;
                                } else {
                                    edge_props.push((k, v));
                                }
                            }
                            stats.connections_created += 1;
                        }
                    }
                    ConflictHandling::Preserve => {
                        let interned_props = conn.properties;
                        if let Some(EdgeData {
                            properties: edge_props,
                            ..
                        }) = GraphWrite::edge_weight_mut(&mut graph.graph, edge_idx)
                        {
                            for (k, v) in interned_props {
                                if !edge_props.iter().any(|(ek, _)| *ek == k) {
                                    edge_props.push((k, v));
                                }
                            }
                            stats.connections_created += 1;
                        }
                    }
                    ConflictHandling::Sum => {
                        let interned_props = conn.properties;
                        if let Some(EdgeData {
                            properties: edge_props,
                            ..
                        }) = GraphWrite::edge_weight_mut(&mut graph.graph, edge_idx)
                        {
                            for (k, v) in interned_props {
                                if let Some((_, existing)) =
                                    edge_props.iter_mut().find(|(ek, _)| *ek == k)
                                {
                                    *existing = sum_values(existing, &v);
                                } else {
                                    edge_props.push((k, v));
                                }
                            }
                            stats.connections_created += 1;
                        }
                    }
                }
            } else {
                let edge_data = EdgeData::new_interned(conn_type_key, conn.properties);
                let new_id = GraphWrite::add_edge(
                    &mut graph.graph,
                    conn.source_idx,
                    conn.target_idx,
                    edge_data,
                );
                // Within-chunk dedup: later iterations targeting the same
                // (src, tgt) resolve to this edge via Update/Preserve/Sum.
                // Skipped for the initial-load path, whose lookup stays empty.
                if !self.skip_existence_check {
                    existing_lookup.insert((conn.source_idx, conn.target_idx), new_id);
                }
                stats.connections_created += 1;
            }
        }

        graph.invalidate_edge_type_counts_cache();

        self.metrics.processing_time += start.elapsed().as_secs_f64();
        self.metrics.batch_count += 1;
        self.metrics.memory_used = self.connections.capacity();

        stats.properties_tracked = self.schema_properties.len();
        Ok(stats)
    }

    pub fn execute(
        mut self,
        graph: &mut DirGraph,
        connection_type: String,
    ) -> Result<(ConnectionBatchStats, BatchMetrics), String> {
        graph.register_connection_type(connection_type.clone());

        let mut total_stats = self.accumulated_stats;

        match self.batch_type {
            BatchType::Small | BatchType::Medium => {
                let stats = self.flush_chunk(graph, &connection_type)?;
                total_stats.combine(&stats);
            }
            BatchType::Large => {
                if !self.connections.is_empty() {
                    let stats = self.flush_chunk(graph, &connection_type)?;
                    total_stats.combine(&stats);
                }
            }
        }

        Ok((total_stats, self.metrics))
    }

    /// The interned keys of every property any queued edge actually carried.
    /// Callers resolve them through `graph.interner` when they need names.
    pub fn get_schema_properties(&self) -> &HashSet<InternedKey> {
        &self.schema_properties
    }

    /// Resolved property → type-name map for schema registration: every
    /// tracked key, typed by its first concrete observation, "Unknown" only
    /// for keys never seen with a non-null value.
    pub fn schema_property_types(&self, graph: &DirGraph) -> HashMap<String, String> {
        self.schema_properties
            .iter()
            .map(|key| {
                (
                    graph.interner.resolve(*key).to_string(),
                    self.schema_property_types
                        .get(key)
                        .map(|t| (*t).to_string())
                        .unwrap_or_else(|| "Unknown".to_string()),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_values_int_int() {
        assert_eq!(
            sum_values(&Value::Int64(10), &Value::Int64(5)),
            Value::Int64(15)
        );
    }

    #[test]
    fn test_sum_values_int_negative() {
        assert_eq!(
            sum_values(&Value::Int64(10), &Value::Int64(-3)),
            Value::Int64(7)
        );
    }

    #[test]
    fn test_sum_values_float_float() {
        match sum_values(&Value::Float64(1.5), &Value::Float64(2.5)) {
            Value::Float64(v) => assert!((v - 4.0).abs() < 1e-10),
            other => panic!("Expected Float64, got {:?}", other),
        }
    }

    #[test]
    fn test_sum_values_int_float_promotion() {
        match sum_values(&Value::Int64(10), &Value::Float64(2.5)) {
            Value::Float64(v) => assert!((v - 12.5).abs() < 1e-10),
            other => panic!("Expected Float64, got {:?}", other),
        }
    }

    #[test]
    fn test_sum_values_float_int_promotion() {
        match sum_values(&Value::Float64(3.5), &Value::Int64(2)) {
            Value::Float64(v) => assert!((v - 5.5).abs() < 1e-10),
            other => panic!("Expected Float64, got {:?}", other),
        }
    }

    #[test]
    fn test_sum_values_non_numeric_overwrites() {
        assert_eq!(
            sum_values(&Value::String("old".into()), &Value::String("new".into())),
            Value::String("new".into()),
        );
    }

    #[test]
    fn test_sum_values_null_cases() {
        assert_eq!(sum_values(&Value::Null, &Value::Int64(5)), Value::Int64(5));
        assert_eq!(sum_values(&Value::Int64(5), &Value::Null), Value::Null);
    }
}

/// The WAL payload a mapped/disk `add_nodes` produces must scale with the
/// number of rows written, not with the number of rows the type already holds.
///
/// Regression guard for the quadratic-amplification bug: the columnar
/// detach/reattach sweeps in [`BatchProcessor::detach_columnar_stores`] and
/// [`BatchProcessor::reattach_columnar_stores`] used to touch every existing
/// node of the type, once per `LARGE_BATCH_CHUNK_SIZE` chunk. Through the
/// *recorded* `node_weight_mut` that was `O(n²/chunk)` `RawOp::UpsertNode`s
/// for an `n`-row append, each resolving to a full property-map clone.
///
/// These are deliberately **counting** assertions (WAL ops, then WAL bytes),
/// not timing ones: they need no idle machine, have no `min`-vs-`mean`
/// ambiguity, and fail deterministically on any reintroduction of a recorded
/// borrow in either sweep.
#[cfg(test)]
mod wal_amplification_tests {
    use crate::datatypes::{DataFrame, Value};
    use crate::graph::mutation::maintain::add_nodes;
    use crate::graph::schema::GraphBackend;
    use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};
    use crate::graph::storage::recording::{resolve_ops, RecordingGraph};
    use crate::graph::storage::GraphRead;
    use crate::graph::wal::{append_frame, WalFrame};

    /// Append `n` rows to a fresh mapped graph through a recording backend and
    /// return `(number of resolved WAL ops, encoded frame bytes)`.
    fn wal_cost_of_appending(n: i64) -> (usize, usize) {
        let mut dir = new_dir_graph_in_mode(StorageMode::Mapped, None).expect("mapped graph");
        // Wrap the mapped backend exactly as `setup_durable` does, so the
        // capture seam under test is the real one.
        let inner = std::mem::replace(&mut dir.graph, GraphBackend::new());
        dir.graph = GraphBackend::Recording(Box::new(RecordingGraph::new(inner)));
        assert!(
            dir.graph.is_mapped(),
            "the sweeps under test are mapped-only"
        );

        let columns = vec!["id".to_string(), "name".to_string(), "dept".to_string()];
        let rows: Vec<Vec<Value>> = (0..n)
            .map(|i| {
                vec![
                    Value::Int64(i),
                    Value::String(format!("person-{i}")),
                    Value::String("engineering".to_string()),
                ]
            })
            .collect();
        let df = DataFrame::from_cypher_rows(columns, rows).expect("dataframe");

        add_nodes(
            &mut dir,
            df,
            "Person".to_string(),
            "id".to_string(),
            Some("name".to_string()),
            None,
        )
        .expect("add_nodes");

        let raw = match &mut dir.graph {
            GraphBackend::Recording(rg) => rg.take_ops(),
            _ => unreachable!("wrapped in Recording above"),
        };
        let ops = resolve_ops(&raw, &dir.graph, &dir.interner, |idx| {
            dir.secondary_label_names(idx)
        });
        let op_count = ops.len();

        let mut encoded = Vec::new();
        append_frame(&mut encoded, &WalFrame { lsn: 1, ops }).expect("frame encodes");
        (op_count, encoded.len())
    }

    /// One op per row, at every size.
    ///
    /// `n` spans four `LARGE_BATCH_CHUNK_SIZE` chunks, so the amplifying
    /// shape (chunk `k` re-touching the `k * 1000` rows already present) is
    /// exercised. Pre-fix this was 2 ops/row at 1k and 5 ops/row at 4k.
    #[test]
    fn add_nodes_records_exactly_one_wal_op_per_row() {
        for n in [1000_i64, 4000] {
            let (ops, _) = wal_cost_of_appending(n);
            assert_eq!(
                ops, n as usize,
                "a {n}-row mapped append must record exactly {n} WAL ops, got {ops}; \
                 a columnar detach/reattach sweep is recording again"
            );
        }
    }

    /// WAL bytes per row stay flat as the row count grows.
    ///
    /// The tolerance absorbs the genuine per-row growth in the encoding (id
    /// and title widen by a byte or two as values get longer) while staying
    /// far below the pre-fix blow-up, which was ~2.5x over this same span and
    /// unbounded beyond it.
    #[test]
    fn add_nodes_wal_bytes_per_row_are_flat() {
        let (_, small_bytes) = wal_cost_of_appending(1000);
        let (_, large_bytes) = wal_cost_of_appending(4000);
        let small_per_row = small_bytes as f64 / 1000.0;
        let large_per_row = large_bytes as f64 / 4000.0;
        let ratio = large_per_row / small_per_row;
        assert!(
            ratio < 1.15,
            "WAL bytes/row must not grow with graph size: {small_per_row:.1} B/row at 1k rows \
             vs {large_per_row:.1} B/row at 4k rows (ratio {ratio:.2}). A quadratic payload \
             means a columnar sweep is recording one op per pre-existing node again."
        );
    }
}
