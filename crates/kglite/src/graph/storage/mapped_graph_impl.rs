//! `impl MappedGraph` — type-index / property-index build helpers and
//! the columnar-mode `flatten_to_csr` helper used by both index builds.
//!
//! Split out of `storage/mod.rs` to keep that file under its 800-line
//! cap. Lives in a sibling `impl MappedGraph {}` block.

use crate::datatypes::Value;
use crate::graph::schema::{EdgeData, InternedKey, NodeData};
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::stable_graph::StableDiGraph;
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::{flatten_to_csr, MappedGraph, MappedPropertyIndex, MappedTypeIndex};

impl MappedGraph {
    #[inline]
    pub fn new() -> Self {
        Self {
            inner: StableDiGraph::new(),
            type_index: RwLock::new(HashMap::new()),
            property_index: RwLock::new(HashMap::new()),
            global_property_index: RwLock::new(HashMap::new()),
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

    /// Drop the cached property indexes (both per-type and global).
    /// Called by node-mutation paths (`add_node`, `remove_node`,
    /// `node_weight_mut`) since any of those can change the set of
    /// `(value, node_idx)` pairs an index is built from.
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
