// src/graph/batch.rs
use crate::datatypes::Value;
use crate::graph::schema::{
    DirGraph, EdgeData, InternedKey, NodeData, PropertyStorage, PROVISIONAL_KEY,
};
use crate::graph::storage::{GraphRead, GraphWrite};
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::Direction;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

// Constants for batch size optimization
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

// Node Processing
#[derive(Debug)]
pub enum NodeAction {
    Update {
        node_idx: NodeIndex,
        title: Option<Value>, // Changed to Option to indicate if title should be updated
        properties: HashMap<String, Value>,
        conflict_mode: ConflictHandling, // Added conflict mode
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
fn sum_values(existing: &Value, new: &Value) -> Value {
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
    title: Option<Value>, // Changed to Option
    properties: HashMap<String, Value>,
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
    accumulated_stats: BatchStats, // Track stats across intermediate flushes
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
                    conflict_mode, // Add this field
                });
            }
        }

        // For large batches, flush if we hit capacity
        if let BatchType::Large = self.batch_type {
            if self.creates_interned.len() >= self.capacity {
                let stats = self.flush_chunk(graph)?;
                self.accumulated_stats.combine(&stats); // Accumulate stats from intermediate flushes
            }
        }

        Ok(())
    }

    fn flush_chunk(&mut self, graph: &mut DirGraph) -> Result<BatchStats, String> {
        let start = Instant::now();
        let mut stats = BatchStats::default();
        let mapped = graph.graph.is_mapped() || graph.graph.is_disk();

        // In mapped mode, we use a two-pass approach to avoid O(n²) Arc cloning:
        // Pass 1: detach existing nodes' Arc refs, push all rows into owned ColumnStores
        // Pass 2: wrap stores back in Arc, assign refs to all nodes (old + new)
        //
        // This avoids Arc::make_mut cloning the entire store when existing nodes
        // hold shared references.
        let mut deferred_columnar: Vec<(NodeIndex, String, u32)> = Vec::new();
        // Owned mutable column stores, extracted from Arc to avoid clone-on-write
        let mut owned_stores: HashMap<String, crate::graph::storage::column_store::ColumnStore> =
            HashMap::new();

        if mapped {
            let affected_types: HashSet<String> = self
                .creates_interned
                .iter()
                .map(|c| c.node_type.clone())
                .collect();

            // For each affected type: detach existing nodes and extract the store
            for node_type in &affected_types {
                // Detach existing nodes — record their (NodeIndex, row_id) for pass 2
                if let Some(indices) = graph.type_indices.get(node_type) {
                    for idx in indices.iter() {
                        if let Some(node) = GraphWrite::node_weight_mut(&mut graph.graph, idx) {
                            if let PropertyStorage::Columnar { row_id, .. } = &node.properties {
                                let rid = *row_id;
                                node.properties = PropertyStorage::Map(HashMap::new());
                                deferred_columnar.push((idx, node_type.clone(), rid));
                            }
                        }
                    }
                }
                // Extract the store from Arc (now refcount=1, so try_unwrap succeeds)
                if let Some(arc_store) = graph.column_stores.remove(node_type) {
                    let store = Arc::try_unwrap(arc_store).unwrap_or_else(|a| (*a).clone());
                    owned_stores.insert(node_type.clone(), store);
                }
            }
        }

        // Process pre-interned creates (fast path — no string interning needed)
        for creation in self.creates_interned.drain(..) {
            let type_key = graph.interner.get_or_intern(&creation.node_type);

            let mut node_data = if mapped {
                NodeData::new_preinterned(
                    creation.id,
                    creation.title,
                    type_key,
                    creation.properties,
                )
            } else {
                let schema: Option<Arc<_>> = graph.type_schemas.get(&creation.node_type).cloned();
                if let Some(ref ts) = schema {
                    NodeData::new_compact_preinterned(
                        creation.id,
                        creation.title,
                        type_key,
                        creation.properties,
                        ts,
                    )
                } else {
                    NodeData::new_preinterned(
                        creation.id,
                        creation.title,
                        type_key,
                        creation.properties,
                    )
                }
            };

            // Mapped mode: push into owned ColumnStore (pass 1)
            let mapped_row_id = if mapped {
                let interned_props = node_data
                    .properties
                    .drain_to_interned_pairs(&graph.interner);
                let store = owned_stores
                    .entry(creation.node_type.clone())
                    .or_insert_with(|| {
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
                        crate::graph::storage::column_store::ColumnStore::new(
                            schema,
                            &meta,
                            &graph.interner,
                        )
                    });
                // Extend columns if schema grew (new columns in this batch)
                let current_schema = graph.type_schemas.get(&creation.node_type).cloned();
                if let Some(ref cs) = current_schema {
                    if store.schema().len() < cs.len() {
                        let meta = graph
                            .node_type_metadata
                            .get(&creation.node_type)
                            .cloned()
                            .unwrap_or_default();
                        let old_store = std::mem::replace(
                            store,
                            crate::graph::storage::column_store::ColumnStore::new(
                                cs.clone(),
                                &meta,
                                &graph.interner,
                            ),
                        );
                        for rid in 0..old_store.row_count() {
                            // Always push id/title — use Null as fallback to keep
                            // columns in sync with row_count
                            store.push_id(&old_store.get_id(rid).unwrap_or(Value::Null));
                            store.push_title(&old_store.get_title(rid).unwrap_or(Value::Null));
                            let props = old_store.row_properties(rid);
                            store.push_row(&props);
                        }
                    }
                }
                store.push_id(&node_data.id);
                store.push_title(&node_data.title);
                let row_id = store.push_row(&interned_props);
                node_data.id = Value::Null;
                node_data.title = Value::Null;
                node_data.properties = PropertyStorage::Map(HashMap::new());
                Some(row_id)
            } else {
                None
            };

            let node_idx = GraphWrite::add_node(&mut graph.graph, node_data);

            if let Some(row_id) = mapped_row_id {
                deferred_columnar.push((node_idx, creation.node_type.clone(), row_id));
            }

            graph
                .type_indices
                .entry_or_default(creation.node_type)
                .push(node_idx);
            // id_indices is intentionally NOT updated incrementally here.
            // Writing into entry_or_default before any prior `build_id_index`
            // call would create a partial entry that subsequent lookups
            // would trust as complete (build_id_index short-circuits when
            // an entry exists). Leave id_indices alone; `maintain::add_nodes`
            // invalidates the type's entry at return time so the next
            // lookup rebuilds from `type_indices` (the source of truth).
            stats.creates += 1;
        }

        // Mapped mode pass 2: wrap owned stores in Arc, assign refs to all nodes.
        if !deferred_columnar.is_empty() {
            // Put owned stores back into graph.column_stores as Arcs
            for (node_type, store) in owned_stores {
                graph.column_stores.insert(node_type, Arc::new(store));
            }
            // Assign Arc refs to all nodes (existing + newly created).
            // For disk mode, also update the DiskNodeSlot.row_id directly:
            // node_weight_mut() materializes into an arena that gets cleared on
            // the next call, so the property assignment alone doesn't persist
            // the per-type row_id back to the slot. Without this, slot.row_id
            // keeps the slot-index value assigned by add_node().
            for (node_idx, node_type, row_id) in deferred_columnar {
                let arc_store = graph.column_stores.get(&node_type).unwrap().clone();
                if let Some(node) = GraphWrite::node_weight_mut(&mut graph.graph, node_idx) {
                    node.properties = PropertyStorage::Columnar {
                        store: arc_store,
                        row_id,
                    };
                }
                GraphWrite::update_row_id(&mut graph.graph, node_idx, row_id);
            }
            // 0.9.2: disk-side reads (node_weight, get_node_id, get_node_title)
            // resolve via `disk_graph.column_stores`, NOT the DirGraph-level
            // `graph.column_stores` we just populated. Without this sync, a
            // freshly-built disk graph from `from_blueprint` (or any
            // multi-create add_nodes path) has empty disk-side column_stores
            // — every property read returns Null until save+reload bridges
            // them. Existing sync site in the update path
            // (`disk_updates_applied`) only fires when an UPDATE is in the
            // chunk, never on creates-only.
            if GraphRead::is_disk(&graph.graph) {
                graph.sync_disk_column_stores();
            }
        }

        // Process updates in current chunk.
        //
        // Disk vs memory/mapped split (Phase 5 xfail fix):
        // - Memory / mapped: `node_weight_mut` returns a live `&mut NodeData`.
        //   `node.properties.insert` does `Arc::make_mut(store)` which clones
        //   the store onto the node and mutates the clone. Reads go through
        //   the node's own properties Arc → see updates immediately.
        // - Disk: `node_weight_mut` materialises NodeData into an arena that
        //   `clear_arenas` drops on the next `&mut self` call. Mutations via
        //   the arena never reach `dg.column_stores`, which is where
        //   `DiskGraph::get_node_property` reads from. To fix, disk updates
        //   mutate `graph.column_stores` directly via `Arc::make_mut` and
        //   then re-sync to `dg.column_stores` at the end of the loop.
        //   O(types) clones per chunk instead of the broken O(rows) pattern.
        let is_disk = GraphRead::is_disk(&graph.graph);
        let mut disk_updates_applied = false;
        // Promotion: a real node-row upsert clears the `_provisional`
        // stub marker. Interned once — only the Update/Sum arms use it.
        let provisional_key = graph.interner.get_or_intern(PROVISIONAL_KEY);

        for update in self.updates.drain(..) {
            if update.conflict_mode == ConflictHandling::Skip {
                continue;
            }

            // Pre-intern property keys before borrowing graph.graph mutably.
            let interned_props: Vec<(InternedKey, Value)> = update
                .properties
                .into_iter()
                .map(|(k, v)| {
                    let key = graph.interner.get_or_intern(&k);
                    (key, v)
                })
                .collect();

            if is_disk {
                // Resolve (type_name, row_id) from the disk slot.
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

                let Some(arc_store) = graph.column_stores.get_mut(&type_name) else {
                    continue;
                };
                let store = Arc::make_mut(arc_store);
                match update.conflict_mode {
                    ConflictHandling::Skip => unreachable!(),
                    ConflictHandling::Replace => {
                        if let Some(new_title) = update.title {
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
                        for (k, v) in interned_props {
                            store.set(row_id, k, &v, None);
                        }
                    }
                    ConflictHandling::Update | ConflictHandling::Sum => {
                        if let Some(new_title) = update.title {
                            store.set_title(row_id, &new_title);
                        }
                        for (k, v) in interned_props {
                            store.set(row_id, k, &v, None);
                        }
                        // Promote: a real-row upsert clears the stub marker.
                        if store.get(row_id, provisional_key).is_some() {
                            store.set(row_id, provisional_key, &Value::Null, None);
                        }
                    }
                    ConflictHandling::Preserve => {
                        if let Some(new_title) = update.title {
                            let cur = store.get_title(row_id).unwrap_or(Value::Null);
                            if matches!(cur, Value::Null) {
                                store.set_title(row_id, &new_title);
                            }
                        }
                        for (k, v) in interned_props {
                            if store.get(row_id, k).is_none() {
                                store.set(row_id, k, &v, None);
                            }
                        }
                    }
                }
                disk_updates_applied = true;
                stats.updates += 1;
            } else if let Some(node) =
                GraphWrite::node_weight_mut(&mut graph.graph, update.node_idx)
            {
                match update.conflict_mode {
                    ConflictHandling::Skip => unreachable!(),
                    ConflictHandling::Replace => {
                        if let Some(new_title) = update.title {
                            node.title = new_title;
                        }
                        node.properties.replace_all(interned_props);
                    }
                    ConflictHandling::Update | ConflictHandling::Sum => {
                        if let Some(new_title) = update.title {
                            node.title = new_title;
                        }
                        for (k, v) in interned_props {
                            node.properties.insert(k, v);
                        }
                        // Promote: a real-row upsert clears the stub marker.
                        if node.properties.get(provisional_key).is_some() {
                            node.properties.insert(provisional_key, Value::Null);
                        }
                    }
                    ConflictHandling::Preserve => {
                        if let Some(new_title) = update.title {
                            if *node.title() == Value::Null {
                                node.title = new_title;
                            }
                        }
                        for (k, v) in interned_props {
                            node.properties.insert_if_absent(k, v);
                        }
                    }
                }
                stats.updates += 1;
            }
        }

        if disk_updates_applied {
            graph.sync_disk_column_stores();
        }

        // Update metrics
        self.metrics.processing_time += start.elapsed().as_secs_f64();
        self.metrics.batch_count += 1;
        self.metrics.memory_used = self.creates_interned.capacity() + self.updates.capacity();

        Ok(stats)
    }

    pub fn execute(mut self, graph: &mut DirGraph) -> Result<(BatchStats, BatchMetrics), String> {
        // Start with accumulated stats from intermediate flushes (for large batches)
        let mut total_stats = self.accumulated_stats;

        match self.batch_type {
            BatchType::Small | BatchType::Medium => {
                // Process in a single batch
                let stats = self.flush_chunk(graph)?;
                total_stats.combine(&stats);
            }
            BatchType::Large => {
                // Process any remaining items
                if !self.creates_interned.is_empty() || !self.updates.is_empty() {
                    let stats = self.flush_chunk(graph)?;
                    total_stats.combine(&stats);
                }
            }
        }

        Ok((total_stats, self.metrics))
    }
}

// Connection Processing
#[derive(Debug)]
struct ConnectionCreation {
    source_idx: NodeIndex,
    target_idx: NodeIndex,
    properties: HashMap<String, Value>,
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
    schema_properties: HashSet<String>,
    capacity: usize,
    batch_type: BatchType,
    metrics: BatchMetrics,
    conflict_mode: ConflictHandling,
    accumulated_stats: ConnectionBatchStats, // Track stats across intermediate flushes
    skip_existence_check: bool,              // Skip find_edge() on initial load
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
            capacity,
            batch_type,
            metrics: BatchMetrics::default(),
            conflict_mode: ConflictHandling::Update,
            accumulated_stats: ConnectionBatchStats::default(),
            skip_existence_check: false,
        }
    }

    // Add setter for conflict mode
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
        mut properties: HashMap<String, Value>,
        graph: &mut DirGraph,
        connection_type: &str,
    ) -> Result<(), String> {
        // Freshness provenance: stamp `updated_at` when this edge type opted in
        // (single chokepoint for every `add_connections` route; registered into
        // `schema_properties` below so the columnar edge store gets a slot).
        graph.inject_edge_provenance(connection_type, &mut properties);
        // Skip existence check on initial load (no existing edges of this type)
        if !self.skip_existence_check {
            // Check if an edge of the same type already exists between these nodes
            let conn_type_key = graph.interner.get_or_intern(connection_type);
            let existing_edge = graph
                .graph
                .edges_connecting(source_idx, target_idx)
                .find(|e| e.weight().connection_type == conn_type_key)
                .map(|e| e.id());

            // If edge exists and conflict mode is Skip, don't add it
            if existing_edge.is_some() && self.conflict_mode == ConflictHandling::Skip {
                return Ok(());
            }
        }

        // Track property names for schema
        for key in properties.keys() {
            self.schema_properties.insert(key.clone());
        }

        self.connections.push(ConnectionCreation {
            source_idx,
            target_idx,
            properties,
        });

        // For large batches, flush if we hit capacity
        if let BatchType::Large = self.batch_type {
            if self.connections.len() >= self.capacity {
                let stats = self.flush_chunk(graph, connection_type)?;
                self.accumulated_stats.combine(&stats); // Accumulate stats from intermediate flushes
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

        // Pre-intern the connection type for edge type comparison
        let conn_type_key = graph.interner.get_or_intern(connection_type);

        // A1 fix: build a per-flush (source, target) -> edge_id map once,
        // restricted to the chunk's unique source set and this connection
        // type. Replaces the per-edge `edges_connecting().find()` walk —
        // for hub-source fan-out into an *existing* connection type, the
        // old code was O(N * max_degree); this is O(sum_of_unique_source_degrees).
        //
        // The map is mutated as we go: newly-created edges are inserted,
        // Replace-mode edges have their entry updated to the new id, and
        // Update/Preserve/Sum modes leave the id untouched. This
        // preserves the within-chunk dedup semantics of the original
        // `edges_connecting`-per-iteration code: two chunk entries with
        // the same (src, tgt) consolidate onto a single edge instead of
        // creating duplicates.
        //
        // `skip_existence_check` (initial-load fast path) skips both the
        // build and the per-edge lookup entirely — there are no existing
        // edges of this type, and within-chunk consolidation is the
        // responsibility of the caller in that mode.
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

        // Create or update edges in current chunk
        for conn in self.connections.drain(..) {
            // On initial load, skip existence check for performance (no existing edges).
            // Otherwise consult the per-flush lookup map built above.
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
                        // Skip this edge (should already be filtered in add_connection)
                        continue;
                    }
                    ConflictHandling::Replace => {
                        // Remove the existing edge and create a new one
                        GraphWrite::remove_edge(&mut graph.graph, edge_idx);
                        let edge_data = EdgeData::new(
                            connection_type.to_string(),
                            conn.properties,
                            &mut graph.interner,
                        );
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
                        // Update existing edge properties
                        // Pre-intern keys before getting mutable edge reference
                        let interned_props: Vec<(InternedKey, Value)> = conn
                            .properties
                            .into_iter()
                            .map(|(k, v)| {
                                let key = graph.interner.get_or_intern(&k);
                                (key, v)
                            })
                            .collect();
                        if let Some(EdgeData {
                            properties: edge_props,
                            ..
                        }) = GraphWrite::edge_weight_mut(&mut graph.graph, edge_idx)
                        {
                            // Merge properties, preferring new values
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
                        // Update but preserve existing values
                        // Pre-intern keys before getting mutable edge reference
                        let interned_props: Vec<(InternedKey, Value)> = conn
                            .properties
                            .into_iter()
                            .map(|(k, v)| {
                                let key = graph.interner.get_or_intern(&k);
                                (key, v)
                            })
                            .collect();
                        if let Some(EdgeData {
                            properties: edge_props,
                            ..
                        }) = GraphWrite::edge_weight_mut(&mut graph.graph, edge_idx)
                        {
                            // Merge properties, preserving existing values
                            for (k, v) in interned_props {
                                if !edge_props.iter().any(|(ek, _)| *ek == k) {
                                    edge_props.push((k, v));
                                }
                            }
                            stats.connections_created += 1;
                        }
                    }
                    ConflictHandling::Sum => {
                        // Sum numeric properties, overwrite non-numeric
                        let interned_props: Vec<(InternedKey, Value)> = conn
                            .properties
                            .into_iter()
                            .map(|(k, v)| {
                                let key = graph.interner.get_or_intern(&k);
                                (key, v)
                            })
                            .collect();
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
                // Create new edge
                let edge_data = EdgeData::new(
                    connection_type.to_string(),
                    conn.properties,
                    &mut graph.interner,
                );
                let new_id = GraphWrite::add_edge(
                    &mut graph.graph,
                    conn.source_idx,
                    conn.target_idx,
                    edge_data,
                );
                // Within-chunk dedup: later iterations targeting the same
                // (src, tgt) now resolve to this edge via Update/Preserve/Sum.
                // No-op when skip_existence_check is true (the lookup is
                // unused and kept empty for that path).
                if !self.skip_existence_check {
                    existing_lookup.insert((conn.source_idx, conn.target_idx), new_id);
                }
                stats.connections_created += 1;
            }
        }

        // Invalidate edge type count cache after edge mutations
        graph.invalidate_edge_type_counts_cache();

        // Update metrics
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
        // Register connection type for O(1) lookups
        graph.register_connection_type(connection_type.clone());

        // Start with accumulated stats from intermediate flushes (for large batches)
        let mut total_stats = self.accumulated_stats;

        match self.batch_type {
            BatchType::Small | BatchType::Medium => {
                // Process in a single batch
                let stats = self.flush_chunk(graph, &connection_type)?;
                total_stats.combine(&stats);
            }
            BatchType::Large => {
                // Process any remaining items
                if !self.connections.is_empty() {
                    let stats = self.flush_chunk(graph, &connection_type)?;
                    total_stats.combine(&stats);
                }
            }
        }

        Ok((total_stats, self.metrics))
    }

    pub fn get_schema_properties(&self) -> &HashSet<String> {
        &self.schema_properties
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
