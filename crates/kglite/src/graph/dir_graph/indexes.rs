//! Secondary-index management on `DirGraph` — the hash equality, composite,
//! and B-tree range index stores, plus the incremental maintenance the Cypher
//! mutation executor calls after each write.
//!
//! Split out of `mod.rs` to keep it under the god-file LoC ceiling; index
//! management is a self-contained concern with one entry point per index kind.
//!
//! The live stores are `#[serde(skip)]`; `populate_index_keys` snapshots their
//! keys before save and `rebuild_indices_from_keys` replays them on load (both
//! in `mod.rs`, with the other serialization helpers).

use std::borrow::Cow;
use std::collections::HashMap;

use super::{DirGraph, IndexStats};
use crate::datatypes::values::Value;
use crate::graph::schema::{CompositeIndexKey, CompositeValue, IndexKey, InternedKey};
use crate::graph::storage::backend::GraphBackend;
use crate::graph::storage::undo::BucketId;
use crate::graph::storage::GraphRead;
use petgraph::graph::NodeIndex;

/// One property name, pre-resolved and pre-interned for a per-node read loop.
///
/// Built once by [`DirGraph::property_reader`], then handed to
/// [`DirGraph::read_indexed`] per node, so alias resolution and interning stay
/// out of the loop. Every index and constraint build reads through this so they
/// all cover exactly the value-space `MATCH` consults.
pub(crate) struct PropertyReader {
    /// The canonical name after id-alias / title-alias resolution.
    resolved: String,
    /// `resolved`, interned — the key the property stores are keyed by.
    key: InternedKey,
}

impl DirGraph {
    // ========================================================================
    // Index Management Methods
    // ========================================================================

    /// Pre-resolve and pre-intern `property` for a per-node read loop.
    pub(crate) fn property_reader(&mut self, node_type: &str, property: &str) -> PropertyReader {
        let resolved = self.resolve_alias(node_type, property).to_string();
        let key = self.interner.get_or_intern(&resolved);
        PropertyReader { resolved, key }
    }

    /// Read one property off `node_idx` **the way the matcher reads it**, so an
    /// index or constraint built from this covers the same value-space `MATCH`
    /// consults (`core/pattern_matching/matcher.rs::
    /// node_matches_properties_columnar`). Three concerns that
    /// `NodeData::get_property` does not handle:
    ///
    /// 1. **Alias resolution.** `starId` may be an id-alias for `id`; same for
    ///    title-aliases. Without resolving, we would look up "starId" in
    ///    PropertyStorage and miss the data (stored under "id"). Handled when
    ///    the [`PropertyReader`] is built.
    /// 2. **`id` / `title` are special.** Their values live in `node_slots`
    ///    (disk) / dedicated `NodeData` fields, NOT in `properties`. The
    ///    matcher reads them via `get_node_id` / `get_node_title`; so do we.
    /// 3. **Column-aware reads.** For mapped/disk graphs loaded from `.kgl` the
    ///    values live in a `ColumnStore`, not in the node's
    ///    `PropertyStorage::Map`/`Compact` snapshot. The backend's
    ///    `get_node_property` knows how to read each storage type;
    ///    `NodeData::get_property` reads only the in-memory snapshot and
    ///    silently returns `None` for column-stored values.
    pub(crate) fn read_indexed(&self, reader: &PropertyReader, idx: NodeIndex) -> Option<Value> {
        match reader.resolved.as_str() {
            "id" => self.graph.get_node_id(idx),
            "title" => self.graph.get_node_title(idx),
            _ => self.graph.get_node_property(idx, reader.key),
        }
    }

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

        // Read through the shared `PropertyReader` so this index covers exactly
        // the value-space MATCH consults — see `read_indexed` for why
        // `NodeData::get_property` is not enough.
        let reader = self.property_reader(node_type, property);
        let mut index: HashMap<Value, Vec<NodeIndex>> = HashMap::new();

        if let Some(node_indices) = self.type_indices.get(node_type) {
            for idx in node_indices.iter() {
                if let Some(value) = self.read_indexed(&reader, idx) {
                    index.entry(value).or_default().push(idx);
                }
            }
        }

        let count = index.len();
        self.property_indices.insert(store_key, index);
        count
    }

    /// Install the single-property equality index for `(node_type, property)`,
    /// **routing by storage backend**. Returns `(entries_indexed, persistent)`.
    ///
    /// On the Disk backend this builds the mmap-backed `PropertyIndex` and
    /// reports `persistent = true`; the in-memory `property_indices` HashMap
    /// would need multiple GB of heap for a ~13M-row type and be rebuilt on
    /// every load. Memory and mapped backends build the HashMap via
    /// [`Self::create_index`].
    ///
    /// **Every caller that installs an equality index on behalf of a user must
    /// route through here.** [`Self::create_index`] is the in-memory primitive
    /// and bypasses the disk decision; calling it directly on a disk graph is
    /// the OOM this method exists to avoid. The counterpart
    /// [`Self::drop_index`] already routes internally.
    ///
    /// Public deliberately, and it is the boundary principle that puts it here:
    /// `kglite-py`'s `create_index` and the Cypher `CREATE INDEX` executor need
    /// the identical backend decision, so it is exactly the shape that belongs
    /// in `kglite::api` rather than being written twice per binding.
    pub fn create_property_index_routed(
        &mut self,
        node_type: &str,
        property: &str,
    ) -> Result<(usize, bool), String> {
        if let GraphBackend::Disk(disk) = &mut self.graph {
            let count = disk
                .build_property_index(node_type, property)
                .map_err(|error| error.to_string())?;
            return Ok((count, true));
        }
        Ok((self.create_index(node_type, property), false))
    }

    /// Drop an index on a property for a specific node type.
    /// Returns true if the index existed and was removed.
    pub fn drop_index(&mut self, node_type: &str, property: &str) -> Result<bool, String> {
        if let GraphBackend::Disk(disk) = &mut self.graph {
            return disk
                .drop_property_index(node_type, property)
                .map_err(|error| format!("persistent index removal failed: {error}"));
        }
        let key = (node_type.to_string(), property.to_string());
        Ok(self.property_indices.remove(&key).is_some())
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
        let reader = self.property_reader(node_type, property);
        let mut index: std::collections::BTreeMap<Value, Vec<NodeIndex>> =
            std::collections::BTreeMap::new();

        if let Some(node_indices) = self.type_indices.get(node_type) {
            for idx in node_indices.iter() {
                if let Some(value) = self.read_indexed(&reader, idx) {
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

        // Pre-resolve + pre-intern each property so the per-node loop is
        // store-lookup only. See `read_indexed` for the read-path rationale.
        let readers: Vec<PropertyReader> = properties
            .iter()
            .map(|p| self.property_reader(node_type, p))
            .collect();

        let mut index: HashMap<CompositeValue, Vec<NodeIndex>> = HashMap::new();

        if let Some(node_indices) = self.type_indices.get(node_type) {
            for idx in node_indices.iter() {
                let values: Vec<Value> = readers
                    .iter()
                    .map(|reader| self.read_indexed(reader, idx).unwrap_or(Value::Null))
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
    // Bulk-path Index Refresh
    // ========================================================================

    /// Rebuild every in-memory secondary index that covers `node_type` from
    /// live graph state. Returns the number of indexes rebuilt.
    ///
    /// **Why the bulk paths need this.** `add_nodes` (and the blueprint builder
    /// and stub vivification, which funnel into it) append rows straight through
    /// `GraphWrite::add_node` and deliberately skip the per-write incremental
    /// maintenance the Cypher mutation executor runs
    /// ([`Self::update_property_indices_for_add`]) — the per-node bookkeeping is
    /// exactly the cost the batch path exists to avoid. But
    /// `try_index_lookup` (`core/pattern_matching/matcher.rs`) consults
    /// `property_indices` **unconditionally**, with no version check, so a
    /// stale index does not degrade to a scan: it silently returns the
    /// pre-load candidate set and an indexed `MATCH` drops the freshly loaded
    /// nodes. Rebuilding once at the end of the bulk call is O(nodes-of-type),
    /// the same order as the `build_id_index` rebuild that path already pays,
    /// instead of O(rows) incremental work.
    ///
    /// A no-op — three `is_empty` checks — when the type carries no index,
    /// which is the common bulk-load case.
    ///
    /// Disk-backed persistent `PropertyIndex` stores are deliberately out of
    /// scope here: they never land in `property_indices` (see
    /// [`Self::create_property_index_routed`]), so this helper cannot see them.
    pub(crate) fn refresh_indexes_for_type(&mut self, node_type: &str) -> usize {
        let prop_keys = Self::keys_for_type(self.property_indices.keys(), node_type);
        let range_keys = Self::keys_for_type(self.range_indices.keys(), node_type);
        let comp_keys: Vec<Vec<String>> = self
            .composite_indices
            .keys()
            .filter(|(nt, _)| nt == node_type)
            .map(|(_, props)| props.clone())
            .collect();

        let unique_keys: Vec<Vec<String>> = self
            .unique_indices
            .keys()
            .filter(|(nt, _)| nt == node_type)
            .map(|(_, props)| props.clone())
            .collect();

        let rebuilt = prop_keys.len() + range_keys.len() + comp_keys.len() + unique_keys.len();
        for property in &prop_keys {
            self.create_index(node_type, property);
        }
        for property in &range_keys {
            self.create_range_index(node_type, property);
        }
        for properties in &comp_keys {
            let refs: Vec<&str> = properties.iter().map(String::as_str).collect();
            self.create_composite_index(node_type, &refs);
        }
        // Unique-constraint occupancy is rebuilt from live data rather than
        // maintained per row. The bulk path validates the whole batch *before*
        // it writes anything (`mutation::maintain::add_nodes`), so by the time we
        // get here the data is known conflict-free and a rebuild reproduces the
        // correct occupancy — without threading per-node claims through the
        // batch engine. Rebuilding is also what keeps a bulk *update* that moves
        // a constrained value from stranding the vacated tuple.
        for properties in &unique_keys {
            let (index, _duplicates, _sample) = self.build_unique_index(node_type, properties);
            let key = (node_type.to_string(), properties.clone());
            self.unique_indices.insert(key, index);
        }
        rebuilt
    }

    /// The property names of the `(node_type, property)` index keys belonging to
    /// `node_type`. Collected into an owned `Vec` so the caller can rebuild
    /// while holding `&mut self`.
    fn keys_for_type<'a>(keys: impl Iterator<Item = &'a IndexKey>, node_type: &str) -> Vec<String> {
        keys.filter(|(nt, _)| nt == node_type)
            .map(|(_, property)| property.clone())
            .collect()
    }

    // ========================================================================
    // Statement-rollback capture for user indexes
    // ========================================================================
    //
    // These helpers give the undo journal the *index* half of a statement's
    // writes. Each records a bucket edit together with the position it
    // touched, because bucket order is the row order an un-`ORDER BY`'d
    // indexed `MATCH` returns (`lookup_by_index` hands the bucket `Vec`
    // straight to the matcher) — restoring membership alone would leave a
    // failed statement visible as a reordering.
    //
    // Each is one `Option` check when no checkpoint is capturing, which is
    // every read and every statement outside a mutating one. They must be
    // called *before* the edit they describe: an eviction needs the position
    // while the node is still in the bucket, and an append needs to know
    // whether the bucket existed beforehand.

    /// Whether a statement checkpoint is currently capturing inverse ops.
    #[inline]
    fn capturing_undo(&mut self) -> bool {
        self.graph.undo_journal_mut().is_some()
    }

    /// Journal `node_idx`'s position in a property-index bucket before it
    /// leaves it.
    fn note_property_eviction(&mut self, key: &IndexKey, value: &Value, node_idx: NodeIndex) {
        if !self.capturing_undo() {
            return;
        }
        let pos = self
            .property_indices
            .get(key)
            .and_then(|value_map| value_map.get(value))
            .and_then(|members| members.iter().position(|member| *member == node_idx));
        let Some(pos) = pos else {
            return;
        };
        let bucket = BucketId::PropertyValue {
            key: key.clone(),
            value: value.clone(),
        };
        if let Some(journal) = self.graph.undo_journal_mut() {
            journal.note_bucket_removed(bucket, node_idx, pos);
        }
    }

    /// Range-index counterpart of [`note_property_eviction`](Self::note_property_eviction).
    fn note_range_eviction(&mut self, key: &IndexKey, value: &Value, node_idx: NodeIndex) {
        if !self.capturing_undo() {
            return;
        }
        let pos = self
            .range_indices
            .get(key)
            .and_then(|btree| btree.get(value))
            .and_then(|members| members.iter().position(|member| *member == node_idx));
        let Some(pos) = pos else {
            return;
        };
        let bucket = BucketId::RangeValue {
            key: key.clone(),
            value: value.clone(),
        };
        if let Some(journal) = self.graph.undo_journal_mut() {
            journal.note_bucket_removed(bucket, node_idx, pos);
        }
    }

    /// Composite-index counterpart of
    /// [`note_property_eviction`](Self::note_property_eviction).
    fn note_composite_eviction(
        &mut self,
        key: &CompositeIndexKey,
        value: &CompositeValue,
        node_idx: NodeIndex,
    ) {
        if !self.capturing_undo() {
            return;
        }
        let pos = self
            .composite_indices
            .get(key)
            .and_then(|comp_map| comp_map.get(value))
            .and_then(|members| members.iter().position(|member| *member == node_idx));
        let Some(pos) = pos else {
            return;
        };
        let bucket = BucketId::CompositeTuple {
            key: key.clone(),
            value: value.clone(),
        };
        if let Some(journal) = self.graph.undo_journal_mut() {
            journal.note_bucket_removed(bucket, node_idx, pos);
        }
    }

    /// Journal an append into a property-index bucket, recording whether the
    /// bucket itself comes into existence with it.
    fn note_property_append(&mut self, key: &IndexKey, value: &Value, node_idx: NodeIndex) {
        if !self.capturing_undo() {
            return;
        }
        let bucket_was_new = match self.property_indices.get(key) {
            Some(value_map) => !value_map.contains_key(value),
            // No such index: the caller's append will not happen either.
            None => return,
        };
        let bucket = BucketId::PropertyValue {
            key: key.clone(),
            value: value.clone(),
        };
        if let Some(journal) = self.graph.undo_journal_mut() {
            journal.note_bucket_appended(bucket, node_idx, bucket_was_new);
        }
    }

    /// Range-index counterpart of [`note_property_append`](Self::note_property_append).
    fn note_range_append(&mut self, key: &IndexKey, value: &Value, node_idx: NodeIndex) {
        if !self.capturing_undo() {
            return;
        }
        let bucket_was_new = match self.range_indices.get(key) {
            Some(btree) => !btree.contains_key(value),
            None => return,
        };
        let bucket = BucketId::RangeValue {
            key: key.clone(),
            value: value.clone(),
        };
        if let Some(journal) = self.graph.undo_journal_mut() {
            journal.note_bucket_appended(bucket, node_idx, bucket_was_new);
        }
    }

    /// Composite-index counterpart of
    /// [`note_property_append`](Self::note_property_append).
    fn note_composite_append(
        &mut self,
        key: &CompositeIndexKey,
        value: &CompositeValue,
        node_idx: NodeIndex,
    ) {
        if !self.capturing_undo() {
            return;
        }
        let bucket_was_new = match self.composite_indices.get(key) {
            Some(comp_map) => !comp_map.contains_key(value),
            None => return,
        };
        let bucket = BucketId::CompositeTuple {
            key: key.clone(),
            value: value.clone(),
        };
        if let Some(journal) = self.graph.undo_journal_mut() {
            journal.note_bucket_appended(bucket, node_idx, bucket_was_new);
        }
    }

    // ========================================================================
    // Incremental Index Maintenance (called by Cypher mutations)
    // ========================================================================

    /// Update property, composite, and range indices after a new node is added.
    /// Only updates indices that already exist for this node_type.
    pub fn update_property_indices_for_add(&mut self, node_type: &str, node_idx: NodeIndex) {
        // Collect single-property index updates (immutable borrow of self.graph)
        let prop_updates: Vec<(IndexKey, Value)> = {
            // Disk backend: node_weight materializes into the query arena,
            // which must run under a DiskQueryGuard (arena protocol in
            // disk/graph.rs). No-op on memory/mapped backends.
            let _guard = self.graph.begin_query();
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
            self.note_property_append(key, value, node_idx);
            if let Some(value_map) = self.property_indices.get_mut(key) {
                value_map.entry(value.clone()).or_default().push(node_idx);
            }
            self.note_range_append(key, value, node_idx);
            if let Some(btree) = self.range_indices.get_mut(key) {
                btree.entry(value.clone()).or_default().push(node_idx);
            }
        }

        // Collect composite index updates
        let comp_updates: Vec<(CompositeIndexKey, CompositeValue)> = {
            // Arena guard: see prop_updates block above.
            let _guard = self.graph.begin_query();
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
            self.note_composite_append(&key, &comp_val, node_idx);
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
        // Update hash index. Vacating the old bucket and joining the new one
        // are journalled separately, and each capture has to read the map as
        // it stands just before its own edit — hence the split borrows.
        if let Some(old_val) = old_value {
            self.note_property_eviction(&key, old_val, node_idx);
            if let Some(value_map) = self.property_indices.get_mut(&key) {
                if let Some(indices) = value_map.get_mut(old_val) {
                    indices.retain(|&idx| idx != node_idx);
                    if indices.is_empty() {
                        value_map.remove(old_val);
                    }
                }
            }
        }
        self.note_property_append(&key, new_value, node_idx);
        if let Some(value_map) = self.property_indices.get_mut(&key) {
            value_map
                .entry(new_value.clone())
                .or_default()
                .push(node_idx);
        }
        // Update range index
        if let Some(old_val) = old_value {
            self.note_range_eviction(&key, old_val, node_idx);
            if let Some(btree) = self.range_indices.get_mut(&key) {
                if let Some(indices) = btree.get_mut(old_val) {
                    indices.retain(|&idx| idx != node_idx);
                    if indices.is_empty() {
                        btree.remove(old_val);
                    }
                }
            }
        }
        self.note_range_append(&key, new_value, node_idx);
        if let Some(btree) = self.range_indices.get_mut(&key) {
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
        self.note_property_eviction(&key, old_value, node_idx);
        if let Some(value_map) = self.property_indices.get_mut(&key) {
            if let Some(indices) = value_map.get_mut(old_value) {
                indices.retain(|&idx| idx != node_idx);
                if indices.is_empty() {
                    value_map.remove(old_value);
                }
            }
        }
        self.note_range_eviction(&key, old_value, node_idx);
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

        // Read current node properties once. Arena guard: disk-backend
        // node_weight materializes into the query arena (protocol in
        // disk/graph.rs); no-op on memory/mapped backends.
        let current_props: HashMap<String, Value> = {
            let _guard = self.graph.begin_query();
            match self.graph.node_weight(node_idx) {
                Some(node) => node.properties_cloned(&self.interner),
                None => return,
            }
        };

        for key in comp_keys {
            // The buckets this node currently sits in, read before vacating
            // any of them so each position can be journalled.
            let occupied: Vec<CompositeValue> = match self.composite_indices.get(&key) {
                Some(comp_map) => comp_map
                    .iter()
                    .filter(|(_, members)| members.contains(&node_idx))
                    .map(|(value, _)| value.clone())
                    .collect(),
                None => continue,
            };
            for value in &occupied {
                self.note_composite_eviction(&key, value, node_idx);
            }
            if let Some(comp_map) = self.composite_indices.get_mut(&key) {
                // Vacate exactly the buckets this node was in, and drop only
                // the ones this call emptied. The blanket
                // `retain(|_, v| !v.is_empty())` this replaces also swept away
                // buckets that were already empty before the call — harmless
                // in itself, but it made the edit unrepresentable as an
                // inverse, and "restore the pre-statement graph" includes the
                // empty buckets it had.
                for value in &occupied {
                    let emptied = match comp_map.get_mut(value) {
                        Some(members) => {
                            members.retain(|&idx| idx != node_idx);
                            members.is_empty()
                        }
                        None => false,
                    };
                    if emptied {
                        comp_map.remove(value);
                    }
                }
            }

            // Build new composite value from current properties
            let new_values: Vec<Value> = key
                .1
                .iter()
                .map(|p| current_props.get(p).cloned().unwrap_or(Value::Null))
                .collect();
            if new_values.iter().any(|v| !matches!(v, Value::Null)) {
                let value = CompositeValue(new_values);
                // After the evictions, so `bucket_was_new` reflects the map
                // the append actually lands in.
                self.note_composite_append(&key, &value, node_idx);
                if let Some(comp_map) = self.composite_indices.get_mut(&key) {
                    comp_map.entry(value).or_default().push(node_idx);
                }
            }
        }
    }
}
