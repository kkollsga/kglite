//! `impl MappedGraph` — construction, `Clone`, the statement-scoped undo
//! journal accessors, type-index / property-index build helpers, and the
//! columnar-mode `flatten_to_csr` helper used by both index builds.
//!
//! Split out of `storage/mod.rs` to keep that file under its 800-line
//! cap. Lives in a sibling `impl MappedGraph {}` block.

use crate::datatypes::Value;
use crate::graph::schema::{EdgeData, InternedKey, NodeData};
use crate::graph::storage::slot_mirror::SlotMirror;
use crate::graph::storage::undo::UndoJournal;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::stable_graph::StableDiGraph;
use petgraph::visit::{EdgeIndexable, NodeIndexable};
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::{MappedGraph, MappedPropertyIndex, MappedTypeIndex};

/// Flatten an adjacency map into CSR form: sorted source list, per-source
/// offsets, and the flat edge array. Both columnar index builds below feed
/// through it.
fn flatten_to_csr(
    mut map: HashMap<NodeIndex, Vec<EdgeIndex>>,
) -> (Vec<NodeIndex>, Vec<u32>, Vec<EdgeIndex>) {
    let mut sources: Vec<NodeIndex> = map.keys().copied().collect();
    sources.sort_by_key(|n| n.index());
    let mut offsets: Vec<u32> = Vec::with_capacity(sources.len() + 1);
    let total: usize = map.values().map(|v| v.len()).sum();
    let mut flat: Vec<EdgeIndex> = Vec::with_capacity(total);
    offsets.push(0);
    for src in &sources {
        if let Some(edges) = map.remove(src) {
            flat.extend(edges);
        }
        offsets.push(flat.len() as u32);
    }
    (sources, offsets, flat)
}

impl Clone for MappedGraph {
    fn clone(&self) -> Self {
        // All lazy indexes are derived state; drop them on clone and
        // let the clone rebuild on demand. Avoids `RwLock` clone
        // plumbing.
        Self {
            inner: self.inner.clone(),
            // Owned state, not a lazy index: one `Arc` bump per node type.
            column_stores: self.column_stores.clone(),
            type_index: RwLock::new(HashMap::new()),
            property_index: RwLock::new(HashMap::new()),
            global_property_index: RwLock::new(HashMap::new()),
            // A journal belongs to the statement that opened it, never to a
            // copy of the graph it was recorded against.
            undo: None,
            // Copied, NOT reset — see `MemoryGraph::clone`. `StableDiGraph`
            // clones its free lists, so this is canonical state, not a cache.
            slot_mirror: self.slot_mirror.clone(),
        }
    }
}

impl MappedGraph {
    /// A genuine deep copy of this backend — see
    /// [`MemoryGraph::deep_clone`](crate::graph::storage::MemoryGraph::deep_clone)
    /// for why the deep copy is named rather than left as a bare `.clone()`:
    /// `GraphBackend::Mapped` holds an `Arc<MappedGraph>`, and on the handle
    /// `.clone()` bumps a refcount instead.
    #[inline]
    pub(crate) fn deep_clone(&self) -> Self {
        self.clone()
    }

    #[inline]
    pub fn new() -> Self {
        Self {
            inner: StableDiGraph::new(),
            column_stores: rustc_hash::FxHashMap::default(),
            type_index: RwLock::new(HashMap::new()),
            property_index: RwLock::new(HashMap::new()),
            global_property_index: RwLock::new(HashMap::new()),
            undo: None,
            slot_mirror: SlotMirror::for_empty_graph(),
        }
    }

    /// Install a fresh statement-scoped undo journal, discarding any stale
    /// one (defensive: a journal must never outlive its statement).
    #[inline]
    pub(crate) fn begin_undo(&mut self) {
        self.undo = Some(Box::new(UndoJournal::new()));
    }

    /// Uninstall and return the journal, ending capture.
    #[inline]
    pub(crate) fn take_undo(&mut self) -> Option<Box<UndoJournal>> {
        self.undo.take()
    }

    /// Mutable access to the active journal, for the `DirGraph`-level capture
    /// seam (inverted-index and timeseries edits the backend cannot see).
    #[inline]
    pub(crate) fn undo_journal_mut(&mut self) -> Option<&mut UndoJournal> {
        self.undo.as_deref_mut()
    }

    /// Wrap an existing petgraph, with every derived index empty.
    ///
    /// Safe precisely because all three indexes are *lazy caches* rebuilt on
    /// first query — the mirror of [`invalidate_type_index`] +
    /// [`invalidate_property_index`], which is what a mutation does. Used by
    /// `DirGraph::vacuum`, which rebuilds the petgraph with contiguous
    /// indices and must land the result back in a `Mapped` backend rather
    /// than silently downgrading the graph to heap storage.
    #[inline]
    pub(crate) fn from_graph(inner: StableDiGraph<NodeData, EdgeData>) -> Self {
        let slot_mirror = SlotMirror::for_adopted_graph(
            inner.node_count(),
            inner.node_bound(),
            inner.edge_count(),
            inner.edge_bound(),
        );
        Self {
            inner,
            column_stores: rustc_hash::FxHashMap::default(),
            type_index: RwLock::new(HashMap::new()),
            property_index: RwLock::new(HashMap::new()),
            global_property_index: RwLock::new(HashMap::new()),
            undo: None,
            slot_mirror,
        }
    }

    /// Borrow the inner `StableDiGraph`. Shared with [`MemoryGraph`]
    /// for match arms that need the heap backend's petgraph view.
    #[inline]
    pub fn inner(&self) -> &StableDiGraph<NodeData, EdgeData> {
        &self.inner
    }

    /// Mutable borrow of the inner `StableDiGraph`.
    #[inline]
    pub fn inner_mut(&mut self) -> &mut StableDiGraph<NodeData, EdgeData> {
        &mut self.inner
    }

    /// Drop the cached type index. Called by `GraphWrite` mutation
    /// methods; subsequent typed-edge queries will rebuild the affected
    /// conn_types on first hit.
    #[inline]
    pub(crate) fn invalidate_type_index(&mut self) {
        if let Ok(mut map) = self.type_index.write() {
            map.clear();
        }
    }

    /// The property-write hook `impl_heap_column_writes!` calls — the mapped
    /// half of `MemoryGraph::note_property_write`, which is a no-op.
    ///
    /// The shared property writers reach `inner.node_weight_mut` through a
    /// disjoint-field destructure rather than through
    /// `GraphWrite::node_weight_mut`, so they get none of that method's
    /// invalidation. Without this hook a `SET`/`REMOVE` left the cached block
    /// mapping the *overwritten* value, and the matcher trusts a non-empty
    /// block verbatim — a wrong `MATCH` result, not a slow one.
    #[inline]
    pub(crate) fn note_property_write(&mut self) {
        self.invalidate_property_index();
    }

    /// Drop the cached property indexes (both per-type and global).
    /// Called by node-mutation paths (`add_node`, `remove_node`,
    /// `node_weight_mut`) and by every property writer, since any of those can
    /// change the set of `(value, node_idx)` pairs an index is built from.
    #[inline]
    pub(crate) fn invalidate_property_index(&mut self) {
        if let Ok(mut map) = self.property_index.write() {
            map.clear();
        }
        if let Ok(mut map) = self.global_property_index.write() {
            map.clear();
        }
    }

    /// Fetch or build the per-(node_type, property) property index.
    /// Build cost: O(|nodes_of_type|) on first hit; subsequent queries
    /// on the same `(node_type, property)` return the cached `Arc`.
    pub(crate) fn ensure_property_index(
        &self,
        node_type: &str,
        property: &str,
    ) -> Arc<MappedPropertyIndex> {
        let key = (node_type.to_string(), property.to_string());
        if let Ok(map) = self.property_index.read() {
            if let Some(block) = map.get(&key) {
                return Arc::clone(block);
            }
        }
        let built = Arc::new(self.build_property_index_block(Some(node_type), property));
        let mut map = match self.property_index.write() {
            Ok(m) => m,
            Err(_) => return built,
        };
        let block = map.entry(key).or_insert_with(|| Arc::clone(&built));
        Arc::clone(block)
    }

    /// Fetch or build a cross-type global property index keyed by
    /// property name only. Iterates every alive node; use for
    /// `MATCH (n {prop: val})` with no label.
    pub(crate) fn ensure_global_property_index(&self, property: &str) -> Arc<MappedPropertyIndex> {
        let key = property.to_string();
        if let Ok(map) = self.global_property_index.read() {
            if let Some(block) = map.get(&key) {
                return Arc::clone(block);
            }
        }
        let built = Arc::new(self.build_property_index_block(None, property));
        let mut map = match self.global_property_index.write() {
            Ok(m) => m,
            Err(_) => return built,
        };
        let block = map.entry(key).or_insert_with(|| Arc::clone(&built));
        Arc::clone(block)
    }

    /// Build a property index from the live nodes of `node_type`
    /// (or every node when `node_type` is `None`). Only `Value::String`
    /// values are indexed — mirrors disk's `PropertyIndex` semantics.
    ///
    /// `InternedKey::from_str` is a deterministic FNV hash so we don't
    /// need access to `DirGraph.interner` here; the result matches what
    /// the nodes themselves stored under.
    ///
    /// Alias handling: the `add_nodes` bulk loader moves the
    /// `node_title_field` column into `NodeData.title` (not into
    /// `properties`), and `unique_id_field` into `NodeData.id`. Disk's
    /// per-type build mirrors this by reading the title/id columns
    /// when the requested property matches an alias
    /// (`title` / `label` / `name`, `id` / `nid` / `qid`). We do the
    /// same here so `lookup_by_property_eq("Person", "name", "Alice")`
    /// finds rows whose name was stored as the title.
    ///
    /// **Columnar rows turn the index off.** A columnar node keeps its values
    /// in the type's `ColumnStore`, which this build does not read: `get_value`
    /// answers `None` for `PropertyStorage::Columnar`, and `title()`/`id()`
    /// return the `Null` sentinel the store is authoritative over. Skipping
    /// such a node would publish a block covering only the row-storage half of
    /// a mixed type, and the matcher returns a non-empty block's answer
    /// verbatim — so `MATCH (n:T {p: v})` would miss the columnar rows
    /// outright. Bailing to an empty block instead reports "no index" and the
    /// matcher scans, which is the same behaviour an all-columnar graph
    /// already gets (nothing indexable is found, so the block is empty).
    ///
    /// That bail is also the reason the per-cell store writers
    /// (`GraphWrite::column_store_mut`, i.e. the executor's `columnar_write`)
    /// need no invalidation hook: a *live* block for a type implies the type
    /// has no columnar rows, and a live global block implies the graph has
    /// none, so no store cell a write can reach is inside one.
    fn build_property_index_block(
        &self,
        node_type: Option<&str>,
        property: &str,
    ) -> MappedPropertyIndex {
        use crate::graph::schema::InternedKey;
        let type_key = node_type.map(InternedKey::from_str);
        let prop_key = InternedKey::from_str(property);
        let is_title_alias = matches!(property, "title" | "label" | "name");
        let is_id_alias = matches!(property, "id" | "nid" | "qid");
        let mut entries: Vec<(String, NodeIndex)> = Vec::new();
        for idx in self.inner.node_indices() {
            let Some(nd) = self.inner.node_weight(idx) else {
                continue;
            };
            if let Some(tk) = type_key {
                if nd.node_type != tk {
                    continue;
                }
            }
            if nd.properties.columnar_row_id().is_some() {
                return MappedPropertyIndex::default();
            }
            // Regular property lookup via InternedKey hash.
            if let Some(Value::String(s)) = nd.properties.get_value(prop_key) {
                entries.push((s, idx));
                continue;
            }
            // Title/id aliases: pull from the dedicated slots.
            if is_title_alias {
                if let Value::String(s) = nd.title().into_owned() {
                    entries.push((s, idx));
                    continue;
                }
            }
            if is_id_alias {
                if let Value::String(s) = nd.id().into_owned() {
                    entries.push((s, idx));
                }
            }
        }
        // Sort by (key, node_idx) for parity with disk's layout.
        entries.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.index().cmp(&b.1.index())));
        let (keys, nodes): (Vec<_>, Vec<_>) = entries.into_iter().unzip();
        MappedPropertyIndex { keys, nodes }
    }

    /// Fetch or build the per-conn-type index block.
    ///
    /// Build cost on first hit: O(|E|) — we scan every edge in the
    /// graph filtering by `conn_type`. Subsequent queries on the same
    /// conn_type reuse the `Arc` in amortised O(1). Memory per block is
    /// ~(2 × 4 bytes × |edges_of_type|) + a peer-count HashMap — for
    /// Wikidata P31 on wiki100m that's ~750 k edges = ~18 MB.
    pub(crate) fn ensure_type_index(&self, conn_type: InternedKey) -> Arc<MappedTypeIndex> {
        let key = conn_type.as_u64();
        // Fast path: already built.
        if let Ok(map) = self.type_index.read() {
            if let Some(block) = map.get(&key) {
                return Arc::clone(block);
            }
        }
        // Slow path: build. Another writer might win the race; that's
        // fine — we just discard our build and use theirs.
        let built = Arc::new(self.build_type_index_block(conn_type));
        let mut map = match self.type_index.write() {
            Ok(m) => m,
            Err(_) => return built,
        };
        let block = map.entry(key).or_insert_with(|| Arc::clone(&built));
        Arc::clone(block)
    }

    fn build_type_index_block(&self, conn_type: InternedKey) -> MappedTypeIndex {
        // Per-source and per-target edge lists (grown via Vec<EdgeIndex>).
        let mut out_map: HashMap<NodeIndex, Vec<EdgeIndex>> = HashMap::new();
        let mut in_map: HashMap<NodeIndex, Vec<EdgeIndex>> = HashMap::new();
        let mut out_peer_counts: HashMap<NodeIndex, i64> = HashMap::new();
        let mut in_peer_counts: HashMap<NodeIndex, i64> = HashMap::new();

        for er in self.inner.edge_references() {
            if er.weight().connection_type != conn_type {
                continue;
            }
            let src = er.source();
            let tgt = er.target();
            let ei = er.id();
            out_map.entry(src).or_default().push(ei);
            in_map.entry(tgt).or_default().push(ei);
            // Outgoing dir → peer = target (edges land on target).
            *out_peer_counts.entry(tgt).or_insert(0) += 1;
            // Incoming dir → peer = source (edges originate at source).
            *in_peer_counts.entry(src).or_insert(0) += 1;
        }

        // Materialise CSR arrays sorted by NodeIndex for binary search.
        let (out_sources, out_offsets, out_edges) = flatten_to_csr(out_map);
        let (in_sources, in_offsets, in_edges) = flatten_to_csr(in_map);

        MappedTypeIndex {
            out_sources,
            out_offsets,
            out_edges,
            in_sources,
            in_offsets,
            in_edges,
            out_peer_counts,
            in_peer_counts,
        }
    }
}
