//! Per-backend [`GraphRead`] / [`GraphWrite`] implementations.
//!
//! Each backend (`MemoryGraph`, `MappedGraph`, `DiskGraph`) owns its own
//! trait impls here, so the backends can diverge without re-touching the
//! enum dispatcher. The `impl GraphRead for GraphBackend` in `backend.rs`
//! is a thin per-variant dispatcher delegating to the impls below.
//! `ForkedGraph`'s impls live alongside it in `forked.rs`.

use crate::datatypes::Value;
use crate::graph::core::filtering::str_values_equal;
use crate::graph::core::iterators::{
    GraphEdgeIndices, GraphEdgeReferences, GraphEdges, GraphEdgesConnecting, GraphNeighbors,
    GraphNodeIndices,
};
use crate::graph::schema::{EdgeData, InternedKey, NodeData};
use crate::graph::storage::column_store::ColumnStore;
use crate::graph::storage::disk::csr::TOMBSTONE_EDGE;
use crate::graph::storage::disk::graph::DiskGraph;
use crate::graph::storage::undo::{ColumnarPreImages, ColumnarWrite};
use crate::graph::storage::{GraphRead, GraphWrite, MappedGraph, MemoryGraph};
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::visit::{EdgeIndexable, EdgeRef, IntoEdgeReferences, NodeIndexable};
use petgraph::Direction;
use std::collections::HashMap;
use std::time::Instant;

// ──────────────────────────────────────────────────────────────────────────
// Heap-backed GraphRead / GraphWrite impls
//
// `MemoryGraph` and `MappedGraph` wrap identical `StableDiGraph` today but
// carry distinct type identity for per-backend divergence. The
// `impl_heap_graph_read!` macro emits the shared read body. The write side is
// written out per backend — `MappedGraph` maintains lazy type/property
// indexes and `MemoryGraph` peer counts, so the trait methods diverge — with
// the parts that don't (column-store writes, undo pre-image capture) shared
// through the two macros below.
// ──────────────────────────────────────────────────────────────────────────

/// The edges `StableGraph::remove_node(idx)` is about to free, **in the order
/// petgraph frees them**.
///
/// `remove_node` walks the outgoing adjacency list to exhaustion removing the
/// head each time, then does the same for the incoming list. `edges_directed`
/// iterates each list head-first, so its order is that removal order. A
/// self-loop sits in *both* lists but is unlinked from both by the outgoing
/// pass, so the incoming pass must skip it — that is the `source() != idx`
/// filter, and getting it wrong would push one slot twice and desynchronise
/// every later edge prediction.
///
/// Order is load-bearing because the free list is LIFO: the same set in a
/// different order predicts different indices. `slot_mirror::tests::
/// the_mirror_predicts_every_slot_petgraph_actually_allocates` pins it against
/// real petgraph, self-loop included.
pub(crate) fn freed_edges_for_removal(
    graph: &petgraph::stable_graph::StableDiGraph<NodeData, EdgeData>,
    idx: NodeIndex,
) -> Vec<EdgeIndex> {
    let mut freed: Vec<EdgeIndex> = graph
        .edges_directed(idx, Direction::Outgoing)
        .map(|e| e.id())
        .collect();
    freed.extend(
        graph
            .edges_directed(idx, Direction::Incoming)
            .filter(|e| e.source() != idx)
            .map(|e| e.id()),
    );
    freed
}

macro_rules! impl_heap_graph_read {
    ($ty:ty, is_memory = $is_memory:expr, is_mapped = $is_mapped:expr) => {
        impl GraphRead for $ty {
            type NodeIndicesIter<'a> = GraphNodeIndices<'a>;
            type EdgeIndicesIter<'a> = GraphEdgeIndices<'a>;
            type EdgesIter<'a> = GraphEdges<'a>;
            type EdgeReferencesIter<'a> = GraphEdgeReferences<'a>;
            type EdgesConnectingIter<'a> = GraphEdgesConnecting<'a>;
            type NeighborsIter<'a> = GraphNeighbors<'a>;

            #[inline]
            fn node_count(&self) -> usize {
                self.inner().node_count()
            }

            #[inline]
            fn edge_count(&self) -> usize {
                self.inner().edge_count()
            }

            #[inline]
            fn node_bound(&self) -> usize {
                self.inner().node_bound()
            }

            #[inline]
            fn edge_bound(&self) -> usize {
                self.inner().edge_bound()
            }

            #[inline]
            fn is_memory(&self) -> bool {
                $is_memory
            }

            #[inline]
            fn is_mapped(&self) -> bool {
                $is_mapped
            }

            #[inline]
            fn is_disk(&self) -> bool {
                false
            }

            #[inline]
            fn node_type_of(&self, idx: NodeIndex) -> Option<InternedKey> {
                self.inner().node_weight(idx).map(|nd| nd.node_type)
            }

            // `node_labels_of` is left on its trait default (primary type
            // only): secondary labels live in `DirGraph.secondary_label_index`,
            // so the full list comes from `DirGraph::node_labels`.

            #[inline]
            fn node_weight(&self, idx: NodeIndex) -> Option<&NodeData> {
                self.inner().node_weight(idx)
            }

            // These four resolve through `NodeView` rather than reading
            // `NodeData` directly, so the heap backends have exactly **one**
            // store-resolution point (`GraphRead::node_view`) — the point the
            // ownership tests' poison hook swaps a store behind.
            #[inline]
            fn get_node_property(&self, idx: NodeIndex, key: InternedKey) -> Option<Value> {
                self.node_view(idx).and_then(|v| v.get_value(key))
            }

            #[inline]
            fn get_node_id(&self, idx: NodeIndex) -> Option<Value> {
                self.node_view(idx).map(|v| v.id().into_owned())
            }

            #[inline]
            fn get_node_title(&self, idx: NodeIndex) -> Option<Value> {
                self.node_view(idx).map(|v| v.title().into_owned())
            }

            #[inline]
            fn str_prop_eq(&self, idx: NodeIndex, key: InternedKey, target: &str) -> Option<bool> {
                self.node_view(idx).and_then(|v| v.str_prop_eq(key, target))
            }

            #[inline]
            fn column_store(&self, type_key: InternedKey) -> Option<&std::sync::Arc<ColumnStore>> {
                self.column_stores.get(&type_key)
            }

            fn column_stores_iter(
                &self,
            ) -> Box<dyn Iterator<Item = (InternedKey, &std::sync::Arc<ColumnStore>)> + '_> {
                Box::new(self.column_stores.iter().map(|(k, v)| (*k, v)))
            }

            #[inline]
            fn node_indices(&self) -> GraphNodeIndices<'_> {
                GraphNodeIndices::InMemory(self.inner().node_indices())
            }

            #[inline]
            fn edge_indices(&self) -> GraphEdgeIndices<'_> {
                GraphEdgeIndices::InMemory(self.inner().edge_indices())
            }

            #[inline]
            fn edge_references(&self) -> GraphEdgeReferences<'_> {
                GraphEdgeReferences::InMemory(self.inner().edge_references())
            }

            #[inline]
            fn edge_weights<'a>(&'a self) -> Box<dyn Iterator<Item = &'a EdgeData> + 'a> {
                Box::new(self.inner().edge_weights())
            }

            #[inline]
            fn edges_directed(&self, idx: NodeIndex, dir: Direction) -> GraphEdges<'_> {
                GraphEdges::InMemory(self.inner().edges_directed(idx, dir))
            }

            #[inline]
            fn edges(&self, idx: NodeIndex) -> GraphEdges<'_> {
                GraphEdges::InMemory(self.inner().edges(idx))
            }

            #[inline]
            fn edges_directed_filtered(
                &self,
                idx: NodeIndex,
                dir: Direction,
                _conn_type_filter: Option<InternedKey>,
            ) -> GraphEdges<'_> {
                // Heap backends don't have a pre-filter fast path; callers
                // still post-filter on `connection_type`.
                GraphEdges::InMemory(self.inner().edges_directed(idx, dir))
            }

            #[inline]
            fn edges_connecting(&self, a: NodeIndex, b: NodeIndex) -> GraphEdgesConnecting<'_> {
                GraphEdgesConnecting::InMemory(self.inner().edges_connecting(a, b))
            }

            #[inline]
            fn edge_weight(&self, idx: EdgeIndex) -> Option<&EdgeData> {
                self.inner().edge_weight(idx)
            }

            #[inline]
            fn find_edge(&self, a: NodeIndex, b: NodeIndex) -> Option<EdgeIndex> {
                self.inner().find_edge(a, b)
            }

            #[inline]
            fn edge_endpoints(&self, idx: EdgeIndex) -> Option<(NodeIndex, NodeIndex)> {
                self.inner().edge_endpoints(idx)
            }

            #[inline(always)]
            fn edge_endpoint_keys<'a>(
                &'a self,
            ) -> Box<dyn Iterator<Item = (NodeIndex, NodeIndex, InternedKey)> + 'a> {
                Box::new(self.inner().edge_references().map(|er| {
                    let w = er.weight();
                    (er.source(), er.target(), w.connection_type)
                }))
            }

            #[inline]
            fn neighbors_directed(&self, idx: NodeIndex, dir: Direction) -> GraphNeighbors<'_> {
                GraphNeighbors::InMemory(self.inner().neighbors_directed(idx, dir))
            }

            #[inline]
            fn neighbors_undirected(&self, idx: NodeIndex) -> GraphNeighbors<'_> {
                GraphNeighbors::InMemory(self.inner().neighbors_undirected(idx))
            }

            fn lookup_peer_counts(&self, conn_type: InternedKey) -> Option<HashMap<u32, i64>> {
                let counts = self.ensure_peer_counts(conn_type);
                Some((*counts.by_target).clone())
            }

            fn count_edges_grouped_by_peer(
                &self,
                conn_type: InternedKey,
                dir: Direction,
                _deadline: Option<Instant>,
            ) -> Result<HashMap<u32, i64>, String> {
                let counts = self.ensure_peer_counts(conn_type);
                let selected = match dir {
                    Direction::Outgoing => &counts.by_target,
                    Direction::Incoming => &counts.by_source,
                };
                Ok((**selected).clone())
            }

            fn count_edges_filtered(
                &self,
                node: NodeIndex,
                dir: Direction,
                conn_type: Option<InternedKey>,
                other_node_type: Option<InternedKey>,
                deadline: Option<Instant>,
            ) -> Result<usize, String> {
                let g = self.inner();
                let mut count = 0;
                for (i, edge) in g.edges_directed(node, dir).enumerate() {
                    if i.is_multiple_of(1 << 20) {
                        if let Some(dl) = deadline {
                            if Instant::now() > dl {
                                return Err("Query timed out".to_string());
                            }
                        }
                    }
                    if let Some(ct) = conn_type {
                        if edge.weight().connection_type != ct {
                            continue;
                        }
                    }
                    let other = if dir == Direction::Outgoing {
                        edge.target()
                    } else {
                        edge.source()
                    };
                    if let Some(required_type) = other_node_type {
                        if let Some(nd) = g.node_weight(other) {
                            if nd.node_type != required_type {
                                continue;
                            }
                        } else {
                            continue;
                        }
                    }
                    count += 1;
                }
                Ok(count)
            }

            // iter_peers_filtered / reset_arenas — trait defaults.
        }
    };
}

impl_heap_graph_read!(MemoryGraph, is_memory = true, is_mapped = false);

// ──────────────────────────────────────────────────────────────────────────
// MemoryGraph — GraphWrite, and the statement-scoped undo capture seam.
//
// Every method follows the same shape: capture the inverse of the edit into
// the journal *if one is installed*, then perform the edit. With no journal
// (the steady state, and every read path) the added cost is one `Option`
// discriminant check; the clone-the-pre-image work lives in `#[cold]`
// helpers so it never bloats the inlined hot path.
//
// See `storage/undo.rs` for the replay contract and
// `dir_graph/rollback.rs` for the restore half.
// ──────────────────────────────────────────────────────────────────────────

impl MemoryGraph {
    /// Clone `idx`'s current weight into the journal as its pre-statement
    /// state, unless this statement already captured it.
    #[cold]
    fn capture_node_weight(&mut self, idx: NodeIndex) {
        let inner = &self.inner;
        if let Some(journal) = self.undo.as_deref_mut() {
            journal.note_node_weight(idx, || inner.node_weight(idx).cloned());
        }
    }

    /// Edge counterpart of [`Self::capture_node_weight`].
    #[cold]
    fn capture_edge_weight(&mut self, idx: EdgeIndex) {
        let inner = &self.inner;
        if let Some(journal) = self.undo.as_deref_mut() {
            journal.note_edge_weight(idx, || inner.edge_weight(idx).cloned());
        }
    }

    fn first_incident_edge(&self, idx: NodeIndex) -> Option<EdgeIndex> {
        self.inner
            .edges_directed(idx, Direction::Outgoing)
            .next()
            .or_else(|| self.inner.edges_directed(idx, Direction::Incoming).next())
            .map(|edge| edge.id())
    }

    /// Detach `idx`'s edges one at a time, through the recorded
    /// [`GraphWrite::remove_edge`], so each gets its own journal entry.
    ///
    /// petgraph's `remove_node` drops incident edges in one internal sweep
    /// whose order we neither observe nor control, which would leave reverse
    /// replay unable to hand each edge back its own free-list slot. Doing the
    /// detach ourselves makes the removal order the recorded order. The end
    /// state is identical either way; the Cypher delete path already detaches
    /// explicitly (`mutation::maintain::detach_delete_nodes`), so in practice
    /// this loop finds nothing to do and exists for every *other* caller of
    /// `remove_node`.
    #[cold]
    fn detach_for_journal(&mut self, idx: NodeIndex) {
        while let Some(edge) = self.first_incident_edge(idx) {
            GraphWrite::remove_edge(self, edge);
        }
    }
}

/// Inherent half of `impl_heap_column_writes!` — the undo hook the trait impl
/// cannot host.
macro_rules! impl_heap_pre_image_capture {
    ($ty:ty) => {
        impl $ty {
            /// Journal the pre-image a property write is about to overwrite.
            ///
            /// One hook for both storage shapes, because a caller cannot know
            /// which it is holding:
            ///
            /// - **row storage** → a `NodeData` clone, exactly what
            ///   `node_weight_mut` captures. The property writers that call
            ///   this hook bypass that method (they need `column_stores` at
            ///   the same time), so they capture here or a failed statement
            ///   cannot restore the value — which is what
            ///   `rollback_tests::set_properties` caught when this was missing.
            /// - **columnar** → the prior value of each cell `write` names, as
            ///   `UndoEntry::ColumnarCell` (plus one
            ///   `UndoEntry::ColumnarSchemaGrown` when the write introduces a
            ///   property the type's schema lacks). There is nothing on the
            ///   node to clone — the value lives in the type's store — and
            ///   capturing the *store* instead would make a one-cell write
            ///   copy every column of the type, which is the cost class the
            ///   journal exists to remove.
            ///
            /// The two-step capture (`ColumnarPreImages::capture` then
            /// `record`) is what keeps the store borrow and the journal borrow
            /// from overlapping; they are sibling fields of this backend.
            #[inline]
            fn capture_property_pre_image(&mut self, idx: NodeIndex, write: ColumnarWrite<'_>) {
                if self.undo.is_none() {
                    return;
                }
                let Some(nd) = self.inner.node_weight(idx) else {
                    return;
                };
                let columnar = nd
                    .properties
                    .columnar_row_id()
                    .map(|row_id| (nd.node_type, row_id));
                match columnar {
                    None => self.capture_node_weight(idx),
                    Some((type_key, row_id)) => {
                        let captured = self
                            .column_stores
                            .get(&type_key)
                            .map(|store| ColumnarPreImages::capture(store, row_id, write));
                        if let (Some(captured), Some(journal)) =
                            (captured, self.undo.as_deref_mut())
                        {
                            captured.record(journal, type_key, row_id);
                        }
                    }
                }
            }

            /// Hide the master row a just-removed node owned, and journal the
            /// flip.
            ///
            /// The columnar half of a node deletion. Without it the row stays
            /// readable — `DETACH DELETE` would leave a ghost that an id lookup
            /// or a `MERGE` could still bind — and the store's live count would
            /// disagree with the type's node count for reasons no consumer
            /// could distinguish from a create.
            ///
            /// The row's values are not cleared: a tombstone hides a row on
            /// every read surface, and leaving the bytes in place is what makes
            /// [`UndoEntry::ColumnarTombstone`] a flag flip rather than a row
            /// pre-image. Consolidation (`enable_columnar`, `vacuum`) is what
            /// finally reclaims them, so they never reach a saved file.
            #[inline]
            fn tombstone_removed_row(&mut self, removed: &NodeData) {
                let Some(row_id) = removed.properties.columnar_row_id() else {
                    return;
                };
                let type_key = removed.node_type;
                let Some(store) = self.column_stores.get_mut(&type_key) else {
                    return;
                };
                std::sync::Arc::make_mut(store).tombstone(row_id);
                if let Some(journal) = self.undo.as_deref_mut() {
                    journal.note_columnar_tombstone(type_key, row_id);
                }
            }
        }
    };
}

impl_heap_pre_image_capture!(MemoryGraph);
impl_heap_pre_image_capture!(MappedGraph);

/// Column-store ownership + node-property writes, shared by the two heap
/// backends.
///
/// A columnar node has no store handle, so a property write needs the store and
/// the node's `row_id` at the same time. Both live on `self`, in different
/// fields, so the disjoint-field borrow below is what makes this expressible at
/// all — and is the reason the property writers sit on the backend rather than
/// on `&mut NodeData`.
///
/// That destructure is also why every writer opens with `note_property_write`:
/// reaching `inner.node_weight_mut` through the fields skips
/// [`GraphWrite::node_weight_mut`] and therefore skips the lazy-index
/// invalidation that method owes. The hook is per-backend — a no-op on
/// `MemoryGraph`, which derives no such index, and
/// `invalidate_property_index` on `MappedGraph`.
macro_rules! impl_heap_column_writes {
    () => {
        #[inline]
        fn install_column_store(
            &mut self,
            type_key: InternedKey,
            store: std::sync::Arc<ColumnStore>,
        ) {
            self.column_stores.insert(type_key, store);
        }

        #[inline]
        fn column_store_mut(
            &mut self,
            type_key: InternedKey,
        ) -> Option<&mut std::sync::Arc<ColumnStore>> {
            self.column_stores.get_mut(&type_key)
        }

        #[inline]
        fn take_column_store(
            &mut self,
            type_key: InternedKey,
        ) -> Option<std::sync::Arc<ColumnStore>> {
            self.column_stores.remove(&type_key)
        }

        #[inline]
        fn clear_column_stores(&mut self) {
            self.column_stores.clear();
        }

        /// Write the node's title through its store's reserved `__title__`
        /// column, leaving the inline field on its `Null` sentinel.
        ///
        /// The inline field is *not* also written: two copies of a title is
        /// what `enable_columnar`'s drift check existed to reconcile, and
        /// keeping the store authoritative is what lets a save skip the
        /// rebuild. A node with no store for its type (row storage, or a type
        /// whose store was dropped) falls back to the inline write.
        fn set_node_title(&mut self, idx: NodeIndex, value: Value) {
            self.note_property_write();
            let columnar = self.inner.node_weight(idx).and_then(|nd| {
                nd.properties
                    .columnar_row_id()
                    .map(|row_id| (nd.node_type, row_id))
            });
            let Some((type_key, row_id)) = columnar else {
                if let Some(nd) = self.inner.node_weight_mut(idx) {
                    nd.title = value;
                }
                return;
            };
            if !self.column_stores.contains_key(&type_key) {
                if let Some(nd) = self.inner.node_weight_mut(idx) {
                    nd.title = value;
                }
                return;
            }
            if self.undo.is_some() {
                let prior = self
                    .column_stores
                    .get(&type_key)
                    .and_then(|store| store.get_title(row_id));
                if let Some(journal) = self.undo.as_deref_mut() {
                    journal.note_columnar_title(type_key, row_id, prior);
                }
            }
            let wrote = self
                .column_stores
                .get_mut(&type_key)
                .is_some_and(|store| std::sync::Arc::make_mut(store).set_title(row_id, &value));
            if !wrote {
                // The row is out of the title column's range — a store shape
                // that cannot hold the write. Fall back through the recorded,
                // journalled node path rather than dropping it.
                if let Some(nd) = GraphWrite::node_weight_mut(self, idx) {
                    nd.title = value;
                }
            }
        }

        fn set_node_property(&mut self, idx: NodeIndex, key: InternedKey, value: Value) {
            self.note_property_write();
            self.capture_property_pre_image(idx, ColumnarWrite::Cell(key));
            let Self {
                inner,
                column_stores,
                ..
            } = self;
            let Some(nd) = inner.node_weight_mut(idx) else {
                return;
            };
            match nd.properties.columnar_row_id() {
                Some(row_id) => {
                    if let Some(store) = column_stores.get_mut(&nd.node_type) {
                        std::sync::Arc::make_mut(store).set(row_id, key, &value, None);
                    }
                }
                None => nd.properties.insert(key, value),
            }
        }

        fn set_node_property_if_absent(&mut self, idx: NodeIndex, key: InternedKey, value: Value) {
            self.note_property_write();
            self.capture_property_pre_image(idx, ColumnarWrite::Cell(key));
            let Self {
                inner,
                column_stores,
                ..
            } = self;
            let Some(nd) = inner.node_weight_mut(idx) else {
                return;
            };
            match nd.properties.columnar_row_id() {
                Some(row_id) => {
                    if let Some(store) = column_stores.get_mut(&nd.node_type) {
                        if store.get(row_id, key).is_none() {
                            std::sync::Arc::make_mut(store).set(row_id, key, &value, None);
                        }
                    }
                }
                None => nd.properties.insert_if_absent(key, value),
            }
        }

        fn remove_node_property(&mut self, idx: NodeIndex, key: InternedKey) -> Option<Value> {
            self.note_property_write();
            self.capture_property_pre_image(idx, ColumnarWrite::Cell(key));
            let Self {
                inner,
                column_stores,
                ..
            } = self;
            let nd = inner.node_weight_mut(idx)?;
            match nd.properties.columnar_row_id() {
                Some(row_id) => {
                    let store = column_stores.get_mut(&nd.node_type)?;
                    let old = store.get(row_id, key);
                    if old.is_some() {
                        std::sync::Arc::make_mut(store).set(row_id, key, &Value::Null, None);
                    }
                    old
                }
                None => nd.properties.remove(key),
            }
        }

        fn clear_node_property(&mut self, idx: NodeIndex, key: InternedKey) -> Option<Value> {
            self.note_property_write();
            self.capture_property_pre_image(idx, ColumnarWrite::Cell(key));
            let Self {
                inner,
                column_stores,
                ..
            } = self;
            let nd = inner.node_weight_mut(idx)?;
            match nd.properties.columnar_row_id() {
                Some(row_id) => {
                    // Columnar removal *is* a Null write, so `remove` already
                    // has the clear semantics the disk flush needs.
                    let store = column_stores.get_mut(&nd.node_type)?;
                    let old = store.get(row_id, key);
                    std::sync::Arc::make_mut(store).set(row_id, key, &Value::Null, None);
                    old
                }
                None => {
                    let prior = nd.properties.remove(key);
                    nd.properties.insert(key, Value::Null);
                    prior
                }
            }
        }

        fn replace_node_properties(&mut self, idx: NodeIndex, pairs: Vec<(InternedKey, Value)>) {
            // Whole-row shape: every present cell is nulled below, then `pairs`
            // are written, so the pre-image spans both sets.
            let written: Vec<InternedKey> = pairs.iter().map(|(k, _)| *k).collect();
            self.note_property_write();
            self.capture_property_pre_image(idx, ColumnarWrite::ReplaceRow(&written));
            let Self {
                inner,
                column_stores,
                ..
            } = self;
            let Some(nd) = inner.node_weight_mut(idx) else {
                return;
            };
            match nd.properties.columnar_row_id() {
                Some(row_id) => {
                    let Some(store) = column_stores.get_mut(&nd.node_type) else {
                        return;
                    };
                    let st = std::sync::Arc::make_mut(store);
                    let existing: Vec<_> = st
                        .row_properties(row_id)
                        .into_iter()
                        .map(|(k, _)| k)
                        .collect();
                    for k in existing {
                        st.set(row_id, k, &Value::Null, None);
                    }
                    for (k, v) in pairs {
                        st.set(row_id, k, &v, None);
                    }
                }
                None => nd.properties.replace_all(pairs),
            }
        }
    };
}

impl GraphWrite for MemoryGraph {
    impl_heap_column_writes!();

    #[inline]
    fn node_weight_mut(&mut self, idx: NodeIndex) -> Option<&mut NodeData> {
        if self.undo.is_some() {
            self.capture_node_weight(idx);
        }
        self.inner.node_weight_mut(idx)
    }

    /// Bypasses the undo journal as well as the WAL recorder.
    ///
    /// No production call site reaches it today: the sweeps that used to —
    /// `mutation::batch`'s columnar detach/reattach and the executor's
    /// end-of-`SET` handle refresh — are gone, now that a node holds a row id
    /// and no store handle. The override still has to exist, because the trait
    /// default forwards to the *recorded* `node_weight_mut`: any silent
    /// per-node sweep that returns would then clone one `NodeData` pre-image
    /// per node *of the type*, the O(V+E)-per-write cost this journal exists
    /// to remove. `dir_graph::rollback_tests` pins the bypass from the seam.
    ///
    /// Nothing may use this method to skip capture: a caller that changes a
    /// node's *content* silently is unrecoverable on rollback.
    #[inline]
    fn node_weight_mut_silent(&mut self, idx: NodeIndex) -> Option<&mut NodeData> {
        self.inner.node_weight_mut(idx)
    }

    #[inline]
    fn edge_weight_mut(&mut self, idx: EdgeIndex) -> Option<&mut EdgeData> {
        self.invalidate_peer_counts();
        if self.undo.is_some() {
            self.capture_edge_weight(idx);
        }
        self.inner.edge_weight_mut(idx)
    }

    #[inline]
    fn add_node(&mut self, data: NodeData) -> NodeIndex {
        let node_type = data.node_type;
        let bound_before = self.inner.node_bound();
        let idx = self.inner.add_node(data);
        self.slot_mirror.note_node_added(bound_before, idx);
        if let Some(journal) = self.undo.as_deref_mut() {
            journal.note_node_added(idx, node_type);
        }
        idx
    }

    #[inline]
    fn remove_node(&mut self, idx: NodeIndex) -> Option<NodeData> {
        self.invalidate_peer_counts();
        if self.undo.is_some() {
            self.detach_for_journal(idx);
        }
        let freed_edges = freed_edges_for_removal(&self.inner, idx);
        let removed = self.inner.remove_node(idx)?;
        self.slot_mirror
            .note_node_removed(idx, freed_edges.into_iter());
        if let Some(journal) = self.undo.as_deref_mut() {
            journal.note_node_removed(idx, removed.clone());
        }
        self.tombstone_removed_row(&removed);
        Some(removed)
    }

    #[inline]
    fn add_edge(&mut self, a: NodeIndex, b: NodeIndex, data: EdgeData) -> EdgeIndex {
        self.invalidate_peer_counts();
        let bound_before = EdgeIndexable::edge_bound(&self.inner);
        let idx = self.inner.add_edge(a, b, data);
        self.slot_mirror.note_edge_added(bound_before, idx);
        if let Some(journal) = self.undo.as_deref_mut() {
            journal.note_edge_added(idx);
        }
        idx
    }

    #[inline]
    fn remove_edge(&mut self, idx: EdgeIndex) -> Option<EdgeData> {
        self.invalidate_peer_counts();
        let endpoints = self.undo.is_some().then(|| self.inner.edge_endpoints(idx));
        let removed = self.inner.remove_edge(idx)?;
        self.slot_mirror.note_edge_removed(idx);
        if let Some(journal) = self.undo.as_deref_mut() {
            if let Some(Some((src, tgt))) = endpoints {
                journal.note_edge_removed(idx, src, tgt, removed.clone());
            }
        }
        Some(removed)
    }

    // update_row_id — trait default no-op (disk-only).
}

// ──────────────────────────────────────────────────────────────────────────
// MappedGraph — hand-written GraphRead impl with lazy per-conn-type and
// per-property indexes. Delegates most methods to `self.inner` (identical to
// the macro body); the bulk-answer methods —
// `sources_for_conn_type_bounded`, `lookup_peer_counts`,
// `count_edges_grouped_by_peer`, `count_edges_filtered` and the four
// `lookup_by_property_*` — consult an index instead. Per-node
// `edges_directed_filtered` deliberately does not; see its comment below.
// Disk already has these structures as persistent mmap; for mapped we
// rebuild them in RAM on first query.
// ──────────────────────────────────────────────────────────────────────────

impl GraphRead for MappedGraph {
    type NodeIndicesIter<'a> = GraphNodeIndices<'a>;
    type EdgeIndicesIter<'a> = GraphEdgeIndices<'a>;
    type EdgesIter<'a> = GraphEdges<'a>;
    type EdgeReferencesIter<'a> = GraphEdgeReferences<'a>;
    type EdgesConnectingIter<'a> = GraphEdgesConnecting<'a>;
    type NeighborsIter<'a> = GraphNeighbors<'a>;

    #[inline]
    fn column_store(&self, type_key: InternedKey) -> Option<&std::sync::Arc<ColumnStore>> {
        self.column_stores.get(&type_key)
    }

    fn column_stores_iter(
        &self,
    ) -> Box<dyn Iterator<Item = (InternedKey, &std::sync::Arc<ColumnStore>)> + '_> {
        Box::new(self.column_stores.iter().map(|(k, v)| (*k, v)))
    }

    #[inline]
    fn node_count(&self) -> usize {
        self.inner().node_count()
    }
    #[inline]
    fn edge_count(&self) -> usize {
        self.inner().edge_count()
    }
    #[inline]
    fn node_bound(&self) -> usize {
        self.inner().node_bound()
    }
    #[inline]
    fn edge_bound(&self) -> usize {
        self.inner().edge_bound()
    }
    #[inline]
    fn is_memory(&self) -> bool {
        false
    }
    #[inline]
    fn is_mapped(&self) -> bool {
        true
    }
    #[inline]
    fn is_disk(&self) -> bool {
        false
    }
    #[inline]
    fn node_type_of(&self, idx: NodeIndex) -> Option<InternedKey> {
        self.inner().node_weight(idx).map(|nd| nd.node_type)
    }
    // node_labels_of: trait default, as in the heap macro above.

    #[inline]
    fn node_weight(&self, idx: NodeIndex) -> Option<&NodeData> {
        self.inner().node_weight(idx)
    }
    // Through `node_view`, which pairs the node's row id with the store this
    // backend owns. Mapped is the mode where this matters most: its nodes hold
    // no row copy of their properties at all, so reading `nd.properties`
    // directly answers `None` for everything.
    #[inline]
    fn get_node_property(&self, idx: NodeIndex, key: InternedKey) -> Option<Value> {
        self.node_view(idx).and_then(|v| v.get_value(key))
    }
    #[inline]
    fn get_node_id(&self, idx: NodeIndex) -> Option<Value> {
        self.node_view(idx).map(|v| v.id().into_owned())
    }
    #[inline]
    fn get_node_title(&self, idx: NodeIndex) -> Option<Value> {
        self.node_view(idx).map(|v| v.title().into_owned())
    }
    #[inline]
    fn str_prop_eq(&self, idx: NodeIndex, key: InternedKey, target: &str) -> Option<bool> {
        self.node_view(idx).and_then(|v| v.str_prop_eq(key, target))
    }
    #[inline]
    fn node_indices(&self) -> GraphNodeIndices<'_> {
        GraphNodeIndices::InMemory(self.inner().node_indices())
    }
    #[inline]
    fn edge_indices(&self) -> GraphEdgeIndices<'_> {
        GraphEdgeIndices::InMemory(self.inner().edge_indices())
    }
    #[inline]
    fn edge_references(&self) -> GraphEdgeReferences<'_> {
        GraphEdgeReferences::InMemory(self.inner().edge_references())
    }
    #[inline]
    fn edge_weights<'a>(&'a self) -> Box<dyn Iterator<Item = &'a EdgeData> + 'a> {
        Box::new(self.inner().edge_weights())
    }
    #[inline]
    fn edges_directed(&self, idx: NodeIndex, dir: Direction) -> GraphEdges<'_> {
        GraphEdges::InMemory(self.inner().edges_directed(idx, dir))
    }
    #[inline]
    fn edges(&self, idx: NodeIndex) -> GraphEdges<'_> {
        GraphEdges::InMemory(self.inner().edges(idx))
    }

    // Per-node typed edge iteration stays on the bare petgraph scan.
    // The index's lookup overhead (RwLock read + HashMap + Arc clone +
    // binary_search) is ~100 ns per call, which dominates when the
    // caller only needs to check for a single edge's presence. Bulk
    // queries go through `sources_for_conn_type_bounded` /
    // `lookup_peer_counts` / `count_edges_grouped_by_peer` overrides
    // below, which amortise the index cost over many answered
    // questions. Callers still post-filter on `connection_type`.
    #[inline]
    fn edges_directed_filtered(
        &self,
        idx: NodeIndex,
        dir: Direction,
        _conn_type_filter: Option<InternedKey>,
    ) -> GraphEdges<'_> {
        GraphEdges::InMemory(self.inner().edges_directed(idx, dir))
    }

    #[inline]
    fn edges_connecting(&self, a: NodeIndex, b: NodeIndex) -> GraphEdgesConnecting<'_> {
        GraphEdgesConnecting::InMemory(self.inner().edges_connecting(a, b))
    }
    #[inline]
    fn edge_weight(&self, idx: EdgeIndex) -> Option<&EdgeData> {
        self.inner().edge_weight(idx)
    }
    #[inline]
    fn find_edge(&self, a: NodeIndex, b: NodeIndex) -> Option<EdgeIndex> {
        self.inner().find_edge(a, b)
    }
    #[inline]
    fn edge_endpoints(&self, idx: EdgeIndex) -> Option<(NodeIndex, NodeIndex)> {
        self.inner().edge_endpoints(idx)
    }
    #[inline(always)]
    fn edge_endpoint_keys<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = (NodeIndex, NodeIndex, InternedKey)> + 'a> {
        Box::new(self.inner().edge_references().map(|er| {
            let w = er.weight();
            (er.source(), er.target(), w.connection_type)
        }))
    }
    #[inline]
    fn neighbors_directed(&self, idx: NodeIndex, dir: Direction) -> GraphNeighbors<'_> {
        GraphNeighbors::InMemory(self.inner().neighbors_directed(idx, dir))
    }
    #[inline]
    fn neighbors_undirected(&self, idx: NodeIndex) -> GraphNeighbors<'_> {
        GraphNeighbors::InMemory(self.inner().neighbors_undirected(idx))
    }

    fn sources_for_conn_type_bounded(
        &self,
        conn_type: InternedKey,
        max: Option<usize>,
    ) -> Option<Vec<u32>> {
        let block = self.ensure_type_index(conn_type);
        let slice = match max {
            Some(n) => &block.out_sources[..n.min(block.out_sources.len())],
            None => &block.out_sources[..],
        };
        Some(slice.iter().map(|n| n.index() as u32).collect())
    }

    fn lookup_peer_counts(&self, conn_type: InternedKey) -> Option<HashMap<u32, i64>> {
        let block = self.ensure_type_index(conn_type);
        // Callers use Outgoing semantics (peer = target); the disk helper
        // does the same. Incoming-dir callers go through
        // `count_edges_grouped_by_peer` instead.
        Some(
            block
                .out_peer_counts
                .iter()
                .map(|(n, c)| (n.index() as u32, *c))
                .collect(),
        )
    }

    // ─── OVERRIDE: O(distinct-peers) via the index instead of full scan ──
    fn count_edges_grouped_by_peer(
        &self,
        conn_type: InternedKey,
        dir: Direction,
        _deadline: Option<Instant>,
    ) -> Result<HashMap<u32, i64>, String> {
        let block = self.ensure_type_index(conn_type);
        let peer_counts = match dir {
            Direction::Outgoing => &block.out_peer_counts,
            Direction::Incoming => &block.in_peer_counts,
        };
        Ok(peer_counts
            .iter()
            .map(|(n, c)| (n.index() as u32, *c))
            .collect())
    }

    fn count_edges_filtered(
        &self,
        node: NodeIndex,
        dir: Direction,
        conn_type: Option<InternedKey>,
        other_node_type: Option<InternedKey>,
        deadline: Option<Instant>,
    ) -> Result<usize, String> {
        // When conn_type is given, use the index to skip non-matching
        // edges entirely. When absent, fall back to the heap scan used
        // by the macro impl.
        let g = self.inner();
        let mut count = 0usize;
        if let Some(ct) = conn_type {
            let block = self.ensure_type_index(ct);
            let (sources, offsets, edges) = match dir {
                Direction::Outgoing => (&block.out_sources, &block.out_offsets, &block.out_edges),
                Direction::Incoming => (&block.in_sources, &block.in_offsets, &block.in_edges),
            };
            if let Ok(pos) = sources.binary_search_by_key(&node.index(), |n| n.index()) {
                let start = offsets[pos] as usize;
                let end = offsets[pos + 1] as usize;
                for (i, &ei) in edges[start..end].iter().enumerate() {
                    if i.is_multiple_of(1 << 20) {
                        if let Some(dl) = deadline {
                            if Instant::now() > dl {
                                return Err("Query timed out".to_string());
                            }
                        }
                    }
                    if let Some(required_type) = other_node_type {
                        let Some((src, tgt)) = g.edge_endpoints(ei) else {
                            continue;
                        };
                        let other = if dir == Direction::Outgoing { tgt } else { src };
                        if let Some(nd) = g.node_weight(other) {
                            if nd.node_type != required_type {
                                continue;
                            }
                        } else {
                            continue;
                        }
                    }
                    count += 1;
                }
            }
            return Ok(count);
        }
        for (i, edge) in g.edges_directed(node, dir).enumerate() {
            if i.is_multiple_of(1 << 20) {
                if let Some(dl) = deadline {
                    if Instant::now() > dl {
                        return Err("Query timed out".to_string());
                    }
                }
            }
            let other = if dir == Direction::Outgoing {
                edge.target()
            } else {
                edge.source()
            };
            if let Some(required_type) = other_node_type {
                if let Some(nd) = g.node_weight(other) {
                    if nd.node_type != required_type {
                        continue;
                    }
                } else {
                    continue;
                }
            }
            count += 1;
        }
        Ok(count)
    }

    // The mapped property indexes build lazily on first hit — the cost model
    // of disk's auto-built `title` global index, triggered by the query
    // rather than by save.
    fn lookup_by_property_eq(
        &self,
        node_type: &str,
        property: &str,
        value: &str,
    ) -> Option<Vec<NodeIndex>> {
        let block = self.ensure_property_index(node_type, property);
        // Disk's contract: `None` means "no index for this (type, property)",
        // and the matcher tries the next alias (nid→id→qid) or falls through
        // to a full scan; `Some(vec)` means the index covers the pair, empty
        // vec included. So "no string values found" has to report as "no
        // index".
        if block.keys.is_empty() {
            return None;
        }
        Some(block.lookup_eq(value))
    }

    fn lookup_by_property_prefix(
        &self,
        node_type: &str,
        property: &str,
        prefix: &str,
        limit: usize,
    ) -> Option<Vec<NodeIndex>> {
        let block = self.ensure_property_index(node_type, property);
        if block.keys.is_empty() {
            return None;
        }
        Some(block.lookup_prefix(prefix, limit))
    }

    fn lookup_by_property_eq_any_type(
        &self,
        property: &str,
        value: &str,
    ) -> Option<Vec<NodeIndex>> {
        let block = self.ensure_global_property_index(property);
        if block.keys.is_empty() {
            return None;
        }
        Some(block.lookup_eq(value))
    }

    fn lookup_by_property_prefix_any_type(
        &self,
        property: &str,
        prefix: &str,
        limit: usize,
    ) -> Option<Vec<NodeIndex>> {
        let block = self.ensure_global_property_index(property);
        if block.keys.is_empty() {
            return None;
        }
        Some(block.lookup_prefix(prefix, limit))
    }
}

// ──────────────────────────────────────────────────────────────────────────
// MappedGraph GraphWrite — lazy-index invalidation + undo-journal capture
//
// Every edge mutation invalidates the type index; `node_weight_mut`,
// `node_weight_mut_silent`, `add_node` and `remove_node` invalidate the
// property index, since each can change the set of `(value, node_idx)` pairs
// it was built from. The property writers shared from
// `impl_heap_column_writes!` invalidate the property index too, through the
// `note_property_write` hook that macro calls — they bypass
// `node_weight_mut`, so the invalidation cannot ride along with it. They leave
// the type index alone: a property write cannot change an edge's conn_type.
//
// On top of that, the same undo-capture seam `MemoryGraph` carries above, in
// the same shape and for the same reason: `inner` is a heap `StableDiGraph`,
// so every `UndoEntry` variant, all keyed on a petgraph index, is expressible
// here. What `StorageMode::Mapped` changes is where *properties* live
// (mmap-spilled column stores), not the node/edge graph, so the journal
// transfers verbatim — never at the cost of the invalidation the method
// already owed.
// ──────────────────────────────────────────────────────────────────────────

impl MappedGraph {
    /// Clone `idx`'s current weight into the journal as its pre-statement
    /// state, unless this statement already captured it.
    #[cold]
    fn capture_node_weight(&mut self, idx: NodeIndex) {
        let inner = &self.inner;
        if let Some(journal) = self.undo.as_deref_mut() {
            journal.note_node_weight(idx, || inner.node_weight(idx).cloned());
        }
    }

    /// Edge counterpart of [`Self::capture_node_weight`].
    #[cold]
    fn capture_edge_weight(&mut self, idx: EdgeIndex) {
        let inner = &self.inner;
        if let Some(journal) = self.undo.as_deref_mut() {
            journal.note_edge_weight(idx, || inner.edge_weight(idx).cloned());
        }
    }

    fn first_incident_edge(&self, idx: NodeIndex) -> Option<EdgeIndex> {
        self.inner
            .edges_directed(idx, Direction::Outgoing)
            .next()
            .or_else(|| self.inner.edges_directed(idx, Direction::Incoming).next())
            .map(|edge| edge.id())
    }

    /// Detach `idx`'s edges one at a time, through the recorded
    /// [`GraphWrite::remove_edge`], so each gets its own journal entry.
    /// Same order argument as `MemoryGraph::detach_for_journal`.
    #[cold]
    fn detach_for_journal(&mut self, idx: NodeIndex) {
        while let Some(edge) = self.first_incident_edge(idx) {
            GraphWrite::remove_edge(self, edge);
        }
    }
}

impl GraphWrite for MappedGraph {
    impl_heap_column_writes!();

    #[inline]
    fn node_weight_mut(&mut self, idx: NodeIndex) -> Option<&mut NodeData> {
        // Caller may mutate properties — invalidate property index.
        self.invalidate_property_index();
        if self.undo.is_some() {
            self.capture_node_weight(idx);
        }
        self.inner_mut().node_weight_mut(idx)
    }

    /// Bypasses the undo journal as well as the WAL recorder.
    ///
    /// Same standing as `MemoryGraph`'s override above: no production call
    /// site reaches it today, and it exists so that one cannot silently cost
    /// a `NodeData` clone per node *of the type*. Without it `MappedGraph`
    /// inherits the trait default, which forwards to the *recorded*
    /// `node_weight_mut` — precisely the quadratic amplification commit
    /// 3bf9ef00 removed from the WAL, re-created inside the journal where no
    /// WAL-byte or backend-clone guard would see it. Mapped is where that is
    /// easiest to lose: the sweeps that would reach this seam are gated on
    /// `is_mapped() || is_disk()`, so no memory-backed test covers them;
    /// `rollback_tests::journal_invariants::the_mapped_silent_write_path_records_nothing`
    /// is the guard.
    ///
    /// Nothing may use this method to skip capture: a caller that changes a
    /// node's *content* silently is unrecoverable on rollback.
    #[inline]
    fn node_weight_mut_silent(&mut self, idx: NodeIndex) -> Option<&mut NodeData> {
        self.invalidate_property_index();
        self.inner_mut().node_weight_mut(idx)
    }

    #[inline]
    fn edge_weight_mut(&mut self, idx: EdgeIndex) -> Option<&mut EdgeData> {
        // Mutating an edge weight can change its connection_type, which
        // would invalidate the per-conn-type index.
        self.invalidate_type_index();
        if self.undo.is_some() {
            self.capture_edge_weight(idx);
        }
        self.inner_mut().edge_weight_mut(idx)
    }

    #[inline]
    fn add_node(&mut self, data: NodeData) -> NodeIndex {
        self.invalidate_property_index();
        let node_type = data.node_type;
        let bound_before = self.inner.node_bound();
        let idx = self.inner_mut().add_node(data);
        self.slot_mirror.note_node_added(bound_before, idx);
        if let Some(journal) = self.undo.as_deref_mut() {
            journal.note_node_added(idx, node_type);
        }
        idx
    }

    #[inline]
    fn remove_node(&mut self, idx: NodeIndex) -> Option<NodeData> {
        // Removing a node removes its incident edges and any property
        // entries pointing at it.
        self.invalidate_type_index();
        self.invalidate_property_index();
        if self.undo.is_some() {
            self.detach_for_journal(idx);
        }
        let freed_edges = freed_edges_for_removal(&self.inner, idx);
        let removed = self.inner_mut().remove_node(idx)?;
        self.slot_mirror
            .note_node_removed(idx, freed_edges.into_iter());
        if let Some(journal) = self.undo.as_deref_mut() {
            journal.note_node_removed(idx, removed.clone());
        }
        self.tombstone_removed_row(&removed);
        Some(removed)
    }

    #[inline]
    fn add_edge(&mut self, a: NodeIndex, b: NodeIndex, data: EdgeData) -> EdgeIndex {
        self.invalidate_type_index();
        let bound_before = EdgeIndexable::edge_bound(&self.inner);
        let idx = self.inner_mut().add_edge(a, b, data);
        self.slot_mirror.note_edge_added(bound_before, idx);
        if let Some(journal) = self.undo.as_deref_mut() {
            journal.note_edge_added(idx);
        }
        idx
    }

    #[inline]
    fn remove_edge(&mut self, idx: EdgeIndex) -> Option<EdgeData> {
        self.invalidate_type_index();
        let endpoints = self.undo.is_some().then(|| self.inner.edge_endpoints(idx));
        let removed = self.inner_mut().remove_edge(idx)?;
        self.slot_mirror.note_edge_removed(idx);
        if let Some(journal) = self.undo.as_deref_mut() {
            if let Some(Some((src, tgt))) = endpoints {
                journal.note_edge_removed(idx, src, tgt, removed.clone());
            }
        }
        Some(removed)
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Disk-backed GraphRead / GraphWrite impls
//
// Every method here delegates to the corresponding inherent on `DiskGraph`
// (CSR + mmap columns + per-query arenas). Disk-only helpers return
// concrete `Some(...)` values where the trait defaults would return `None`.
// ──────────────────────────────────────────────────────────────────────────

impl GraphRead for DiskGraph {
    type NodeIndicesIter<'a> = GraphNodeIndices<'a>;
    type EdgeIndicesIter<'a> = GraphEdgeIndices<'a>;
    type EdgesIter<'a> = GraphEdges<'a>;
    type EdgeReferencesIter<'a> = GraphEdgeReferences<'a>;
    type EdgesConnectingIter<'a> = GraphEdgesConnecting<'a>;
    type NeighborsIter<'a> = GraphNeighbors<'a>;

    #[inline]
    fn column_store(&self, type_key: InternedKey) -> Option<&std::sync::Arc<ColumnStore>> {
        self.column_stores.get(&type_key)
    }

    fn column_stores_iter(
        &self,
    ) -> Box<dyn Iterator<Item = (InternedKey, &std::sync::Arc<ColumnStore>)> + '_> {
        Box::new(self.column_stores.iter().map(|(k, v)| (*k, v)))
    }

    #[inline]
    fn node_count(&self) -> usize {
        DiskGraph::node_count(self)
    }

    #[inline]
    fn edge_count(&self) -> usize {
        DiskGraph::edge_count(self)
    }

    #[inline]
    fn node_bound(&self) -> usize {
        DiskGraph::node_bound(self)
    }

    /// The disk backend's edges are a frozen CSR generation, not a
    /// free-listed `StableDiGraph`: there is no edge slot to leave behind, so
    /// the bound *is* the count. A disk graph therefore never reports edge
    /// fragmentation, which is correct — it reclaims by publishing a fresh
    /// generation (`save_disk`), not by compacting in place.
    #[inline]
    fn edge_bound(&self) -> usize {
        DiskGraph::edge_count(self)
    }

    #[inline]
    fn is_memory(&self) -> bool {
        false
    }

    #[inline]
    fn is_mapped(&self) -> bool {
        false
    }

    #[inline]
    fn is_disk(&self) -> bool {
        true
    }

    #[inline(always)]
    fn node_type_of(&self, idx: NodeIndex) -> Option<InternedKey> {
        DiskGraph::node_type_of(self, idx)
    }

    #[inline]
    fn node_weight(&self, idx: NodeIndex) -> Option<&NodeData> {
        DiskGraph::node_weight(self, idx)
    }

    #[inline]
    fn get_node_property(&self, idx: NodeIndex, key: InternedKey) -> Option<Value> {
        DiskGraph::get_node_property(self, idx, key)
    }

    #[inline]
    fn get_node_id(&self, idx: NodeIndex) -> Option<Value> {
        DiskGraph::get_node_id(self, idx)
    }

    #[inline]
    fn get_node_title(&self, idx: NodeIndex) -> Option<Value> {
        DiskGraph::get_node_title(self, idx)
    }

    #[inline]
    fn str_prop_eq(&self, idx: NodeIndex, key: InternedKey, target: &str) -> Option<bool> {
        // Disk keeps the allocating route — the zero-alloc win is heap/mapped-specific.
        DiskGraph::get_node_property(self, idx, key)
            .map(|v| matches!(v, Value::String(ref s) if str_values_equal(s, target)))
    }

    #[inline]
    fn node_indices(&self) -> GraphNodeIndices<'_> {
        GraphNodeIndices::Disk(self.node_indices_iter())
    }

    #[inline]
    fn edge_indices(&self) -> GraphEdgeIndices<'_> {
        GraphEdgeIndices::Disk(self.edge_indices_iter())
    }

    #[inline]
    fn edge_references(&self) -> GraphEdgeReferences<'_> {
        GraphEdgeReferences::Disk(self.edge_references_iter())
    }

    #[inline]
    fn edge_weights<'a>(&'a self) -> Box<dyn Iterator<Item = &'a EdgeData> + 'a> {
        self.edge_weights_iter()
    }

    #[inline]
    fn edges_directed(&self, idx: NodeIndex, dir: Direction) -> GraphEdges<'_> {
        GraphEdges::Disk(self.edges_directed_iter(idx, dir))
    }

    #[inline]
    fn edges(&self, idx: NodeIndex) -> GraphEdges<'_> {
        GraphEdges::Disk(self.edges_directed_iter(idx, Direction::Outgoing))
    }

    #[inline]
    fn edges_directed_filtered(
        &self,
        idx: NodeIndex,
        dir: Direction,
        conn_type_filter: Option<InternedKey>,
    ) -> GraphEdges<'_> {
        GraphEdges::Disk(self.edges_directed_filtered_iter(
            idx,
            dir,
            conn_type_filter.map(|k| k.as_u64()),
        ))
    }

    #[inline]
    fn edges_connecting(&self, a: NodeIndex, b: NodeIndex) -> GraphEdgesConnecting<'_> {
        GraphEdgesConnecting::Disk(self.edges_connecting_iter(a, b))
    }

    #[inline]
    fn edge_weight(&self, idx: EdgeIndex) -> Option<&EdgeData> {
        DiskGraph::edge_weight(self, idx)
    }

    #[inline]
    fn find_edge(&self, a: NodeIndex, b: NodeIndex) -> Option<EdgeIndex> {
        DiskGraph::find_edge(self, a, b)
    }

    #[inline]
    fn edge_endpoints(&self, idx: EdgeIndex) -> Option<(NodeIndex, NodeIndex)> {
        self.edge_endpoints_fn(idx)
    }

    #[inline(always)]
    fn edge_endpoint_keys<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = (NodeIndex, NodeIndex, InternedKey)> + 'a> {
        Box::new((0..self.next_edge_idx).filter_map(move |i| {
            let ep = self.edge_endpoint(i as usize);
            if ep.source == TOMBSTONE_EDGE {
                return None;
            }
            Some((
                NodeIndex::new(ep.source as usize),
                NodeIndex::new(ep.target as usize),
                InternedKey::from_u64(ep.connection_type),
            ))
        }))
    }

    #[inline]
    fn neighbors_directed(&self, idx: NodeIndex, dir: Direction) -> GraphNeighbors<'_> {
        GraphNeighbors::Disk(self.neighbors_directed_iter(idx, dir))
    }

    #[inline]
    fn neighbors_undirected(&self, idx: NodeIndex) -> GraphNeighbors<'_> {
        GraphNeighbors::Disk(self.neighbors_undirected_iter(idx))
    }

    #[inline]
    fn sources_for_conn_type_bounded(
        &self,
        conn_type: InternedKey,
        max: Option<usize>,
    ) -> Option<Vec<u32>> {
        DiskGraph::sources_for_conn_type_bounded(self, conn_type.as_u64(), max)
    }

    #[inline]
    fn lookup_peer_counts(&self, conn_type: InternedKey) -> Option<HashMap<u32, i64>> {
        DiskGraph::lookup_peer_counts(self, conn_type.as_u64())
    }

    #[inline]
    fn lookup_by_property_eq(
        &self,
        node_type: &str,
        property: &str,
        value: &str,
    ) -> Option<Vec<NodeIndex>> {
        DiskGraph::lookup_property_eq(self, node_type, property, value)
    }

    #[inline]
    fn lookup_by_property_prefix(
        &self,
        node_type: &str,
        property: &str,
        prefix: &str,
        limit: usize,
    ) -> Option<Vec<NodeIndex>> {
        DiskGraph::lookup_property_prefix(self, node_type, property, prefix, limit)
    }

    #[inline]
    fn lookup_by_property_eq_any_type(
        &self,
        property: &str,
        value: &str,
    ) -> Option<Vec<NodeIndex>> {
        DiskGraph::lookup_global_eq(self, property, value)
    }

    #[inline]
    fn lookup_by_property_prefix_any_type(
        &self,
        property: &str,
        prefix: &str,
        limit: usize,
    ) -> Option<Vec<NodeIndex>> {
        DiskGraph::lookup_global_prefix(self, property, prefix, limit)
    }

    #[inline]
    fn count_edges_grouped_by_peer(
        &self,
        conn_type: InternedKey,
        dir: Direction,
        deadline: Option<Instant>,
    ) -> Result<HashMap<u32, i64>, String> {
        DiskGraph::count_edges_grouped_by_peer(self, conn_type.as_u64(), dir, deadline)
    }

    #[inline]
    fn count_edges_filtered(
        &self,
        node: NodeIndex,
        dir: Direction,
        conn_type: Option<InternedKey>,
        other_node_type: Option<InternedKey>,
        deadline: Option<Instant>,
    ) -> Result<usize, String> {
        DiskGraph::count_edges_filtered(
            self,
            node,
            dir,
            conn_type.map(|k| k.as_u64()),
            other_node_type,
            deadline,
        )
    }

    #[inline]
    fn iter_peers_filtered<'a>(
        &'a self,
        node: NodeIndex,
        dir: Direction,
        conn_type: Option<u64>,
    ) -> Box<dyn Iterator<Item = (NodeIndex, EdgeIndex)> + 'a> {
        Box::new(
            DiskGraph::iter_peers_filtered(self, node, dir, conn_type)
                .into_iter()
                .map(|(peer, edge_idx)| (peer, EdgeIndex::new(edge_idx as usize))),
        )
    }

    #[inline]
    fn reset_arenas(&self) {
        DiskGraph::reset_arenas(self);
    }
}

impl GraphWrite for DiskGraph {
    #[inline]
    fn install_column_store(&mut self, type_key: InternedKey, store: std::sync::Arc<ColumnStore>) {
        self.column_stores.insert(type_key, store);
    }

    #[inline]
    fn column_store_mut(
        &mut self,
        type_key: InternedKey,
    ) -> Option<&mut std::sync::Arc<ColumnStore>> {
        self.column_stores.get_mut(&type_key)
    }

    #[inline]
    fn take_column_store(&mut self, type_key: InternedKey) -> Option<std::sync::Arc<ColumnStore>> {
        self.column_stores.remove(&type_key)
    }

    #[inline]
    fn clear_column_stores(&mut self) {
        self.column_stores.clear();
    }

    // Disk keeps its own write protocol: `node_weight_mut` stages a `Map`-form
    // `NodeData` in `node_mut_cache`, and `flush_node_mut_cache` folds the
    // staged keys into the type's store. Writing straight to `column_stores`
    // here would bypass that staging and lose the write on the next flush, so
    // these five deliberately go through the cache rather than through the
    // store the backend owns.
    fn set_node_property(&mut self, idx: NodeIndex, key: InternedKey, value: Value) {
        if let Some(nd) = GraphWrite::node_weight_mut(self, idx) {
            nd.properties.insert(key, value);
        }
    }

    fn set_node_property_if_absent(&mut self, idx: NodeIndex, key: InternedKey, value: Value) {
        if let Some(nd) = GraphWrite::node_weight_mut(self, idx) {
            nd.properties.insert_if_absent(key, value);
        }
    }

    fn remove_node_property(&mut self, idx: NodeIndex, key: InternedKey) -> Option<Value> {
        // A bare removal from the staged Map leaves the column store's value
        // untouched at flush time; the disk contract is a `Null` write.
        self.clear_node_property(idx, key)
    }

    fn clear_node_property(&mut self, idx: NodeIndex, key: InternedKey) -> Option<Value> {
        let nd = GraphWrite::node_weight_mut(self, idx)?;
        let prior = nd.properties.remove(key);
        nd.properties.insert(key, Value::Null);
        prior
    }

    fn replace_node_properties(&mut self, idx: NodeIndex, pairs: Vec<(InternedKey, Value)>) {
        if let Some(nd) = GraphWrite::node_weight_mut(self, idx) {
            nd.properties.replace_all(pairs);
        }
    }

    #[inline]
    fn node_weight_mut(&mut self, idx: NodeIndex) -> Option<&mut NodeData> {
        DiskGraph::node_weight_mut(self, idx)
    }

    #[inline]
    fn edge_weight_mut(&mut self, idx: EdgeIndex) -> Option<&mut EdgeData> {
        DiskGraph::edge_weight_mut(self, idx)
    }

    #[inline]
    fn add_node(&mut self, data: NodeData) -> NodeIndex {
        DiskGraph::add_node(self, data)
    }

    #[inline]
    fn remove_node(&mut self, idx: NodeIndex) -> Option<NodeData> {
        DiskGraph::remove_node(self, idx)
    }

    #[inline]
    fn add_edge(&mut self, a: NodeIndex, b: NodeIndex, data: EdgeData) -> EdgeIndex {
        DiskGraph::add_edge(self, a, b, data)
    }

    #[inline]
    fn remove_edge(&mut self, idx: EdgeIndex) -> Option<EdgeData> {
        DiskGraph::remove_edge(self, idx)
    }

    #[inline]
    fn update_row_id(&mut self, node_idx: NodeIndex, row_id: u32) {
        DiskGraph::update_row_id(self, node_idx, row_id);
    }

    #[inline]
    fn flush_pending_writes(&mut self) {
        // Drains node_mut_cache + edge_mut_cache into column_stores /
        // edge_properties (the steady-state read sources) and resets
        // the materialization arenas. Idempotent — safe to call when
        // the caches are empty.
        DiskGraph::clear_arenas(self);
    }
}
