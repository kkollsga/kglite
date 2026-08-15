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

use std::collections::HashMap;

use super::index_layer::LayeredIndex;
use super::range_index_layer::LayeredRangeIndex;
use super::{DirGraph, IndexStats};
use crate::datatypes::values::Value;
use crate::graph::schema::{CompositeIndexKey, CompositeValue, IndexKey, InternedKey};
use crate::graph::storage::backend::GraphBackend;
use crate::graph::storage::undo::BucketId;
use crate::graph::storage::GraphRead;
use petgraph::graph::NodeIndex;

#[cfg(test)]
thread_local! {
    /// Incremental-maintenance passes that ran past the "this type carries no
    /// user index" gate since the last reset.
    ///
    /// The fifth member of the oracle family (`BACKEND_CLONE_NODES`,
    /// `JOURNAL_NODE_PRE_IMAGES`, `COLUMN_STORE_CLONES`, `SCHEMA_MAP_FORKS`),
    /// and the only one that can see this cost: maintenance on an index-free
    /// type clones no backend, journals nothing, forks no schema map and
    /// allocates only transiently, so every existing counter reads the same
    /// whether the walk runs or returns immediately. What it cost was a value
    /// read-back, three `String` allocations and four hash misses **per written
    /// row** — ~23% of every `SET` on a graph with no indexes at all.
    static INDEX_MAINTENANCE_PASSES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_index_maintenance_passes() {
    INDEX_MAINTENANCE_PASSES.set(0);
}

/// Incremental index-maintenance passes on this thread since the last reset.
#[cfg(test)]
pub(crate) fn index_maintenance_passes() -> usize {
    INDEX_MAINTENANCE_PASSES.get()
}

/// Count one maintenance pass that got past the no-index gate.
#[inline]
fn note_maintenance_pass() {
    #[cfg(test)]
    INDEX_MAINTENANCE_PASSES.set(INDEX_MAINTENANCE_PASSES.get() + 1);
}

#[cfg(test)]
thread_local! {
    /// Whole-type index rebuilds ([`DirGraph::refresh_indexes_for_type`]) since
    /// the last reset.
    ///
    /// The oracle for the bulk fold: a fold and a rebuild leave the same
    /// *values*, so an equality assertion alone cannot tell which one ran, and
    /// a fold that silently declined would pass every correctness pin in
    /// `maintain::incremental_index_tests` by rebuilding. This is also the only
    /// way to pin the fold-vs-rebuild cost gate, whose whole job is to choose
    /// between two correct paths.
    static TYPE_INDEX_REBUILDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_type_index_rebuilds() {
    TYPE_INDEX_REBUILDS.set(0);
}

/// Whole-type index rebuilds on this thread since the last reset.
#[cfg(test)]
pub(crate) fn type_index_rebuilds() -> usize {
    TYPE_INDEX_REBUILDS.get()
}

/// What one node looked like *before* a bulk batch updated it, as the index
/// fold needs to see it.
///
/// The whole reason `add_nodes` can fold an update instead of rebuilding: the
/// batch overwrites the values in place, so the old bucket a node has to vacate
/// and the unique tuple it has to release are only knowable from a read taken
/// before the write. One read pass serves both — the property indexes and the
/// constraint occupancy — which is why they are captured together rather than
/// by two independent passes over the same rows.
pub(crate) struct UpdatedRowPreImage {
    pub(crate) node_idx: NodeIndex,
    /// Old value of each [`DirGraph::maintained_index_properties`] entry.
    /// `None` where the node carried no value for it.
    indexed: Vec<(String, Option<Value>)>,
    /// The unique tuples the node occupied before the batch. Empty when the
    /// type declares no unique constraint.
    claims: Vec<super::constraints::UniqueClaim>,
}

/// One planned bucket move: what an updated node has to leave, and what it has
/// to join, for one maintained property.
struct UpdateMove {
    node_idx: NodeIndex,
    property: String,
    old: Option<Value>,
    landed: Option<Value>,
}

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
    ///    staged `PropertyStorage::Map` snapshot. The backend's
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
    /// An index on an id-alias / title-alias field (e.g. `add_nodes(df, "Star",
    /// "starId", "title")` makes `starId` the alias for the canonical id) is
    /// keyed by the alias spelling but built from the node's id / title, since
    /// that is where the value lives and what a `MATCH` on that name compares
    /// against — [`Self::read_indexed`] does the resolving, and the incremental
    /// updaters read through the same funnel so they agree with this build.
    /// Such an index is not *required*, though: lookups against id-alias names
    /// route through `lookup_by_id_readonly` in the matcher
    /// (`try_index_lookup`), which uses the auto-maintained per-type
    /// `id_index`, and SET-on-id stays in sync because id mutation updates the
    /// id_index directly.
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
        self.property_indices
            .insert(store_key, LayeredIndex::from(index));
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
            let total_entries: usize = idx.iter().map(|(_, v)| v.len()).sum();
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
        self.range_indices
            .insert(key, LayeredRangeIndex::from(index));
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
        self.composite_indices
            .insert(key, LayeredIndex::from(index));
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
            let total_entries: usize = idx.iter().map(|(_, v)| v.len()).sum();
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

    /// Maintain `node_type`'s secondary indexes for a bulk append that only
    /// *created* rows, by giving each appended node the same per-node
    /// treatment a Cypher `CREATE` gives it — instead of rebuilding every
    /// covering index from every member of the type.
    ///
    /// **Why the rebuild is not good enough.** It is O(nodes-of-type) per
    /// index, per call: measured 2026-08-14 (release), appending ten rows to a
    /// 200k-row type costs 88 µs with no index, 6.1 ms with one property index
    /// and 29.9 ms with two. That is the same shape the id index carried until
    /// [`fold_appended_ids_into_index`](DirGraph::fold_appended_ids_into_index)
    /// removed it, and it would otherwise re-impose it on exactly the types a
    /// query-heavy application indexes.
    ///
    /// **Updated rows** are folded too, from the pre-image
    /// [`Self::capture_update_pre_image`] took before the batch wrote: a row
    /// whose indexed value moved vacates its old bucket and joins the new one,
    /// and a row whose value did not move touches no bucket at all — which is
    /// what keeps the untouched buckets byte-identical to a rebuild's. The
    /// pre-image is also what lets a `UNIQUE` tuple be *released* rather than
    /// re-derived, so a constrained type no longer forces the rebuild either.
    ///
    /// Returns `false` — meaning "rebuild instead" — only when the appended
    /// tail cannot be identified (the type's member vector is shorter than the
    /// batch claims to have created), because the tail *is* how a created row
    /// is mapped back to its node.
    ///
    /// **Bucket order.** For creates it is the rebuild's, exactly: the rebuild
    /// walks the type's members in order, so the appended nodes land at the end
    /// of their value buckets — which is where appending puts them. For an
    /// update that *moves* a node, the node joins the end of its new bucket
    /// where a rebuild would place it in member order; that is the same
    /// divergence a Cypher `SET` has always produced
    /// ([`Self::update_property_indices_for_set`] appends), and it is why the
    /// fold-vs-rebuild pin asserts set equality plus untouched-bucket order
    /// rather than whole-index byte equality.
    pub(crate) fn fold_batch_into_user_indexes(
        &mut self,
        node_type: &str,
        created: usize,
        updated: &[UpdatedRowPreImage],
    ) -> bool {
        let constrained = self.type_has_unique_constraints(node_type);
        if !constrained && !self.type_has_user_indexes(node_type) {
            // Nothing to maintain — and nothing for the rebuild to do either.
            return true;
        }
        let tail = if created == 0 {
            Vec::new()
        } else {
            match self.appended_tail(node_type, created) {
                Some(tail) => tail,
                None => return false,
            }
        };
        // Plan the update arm before touching anything: which rows actually
        // moved is what decides fold-vs-rebuild, and declining is only free
        // while nothing has been mutated.
        let moves = self.plan_update_moves(node_type, updated);
        if !self.folding_moves_beats_a_rebuild(node_type, moves.len())
            || (constrained
                && !self.claiming_beats_a_rebuild(node_type, tail.len() + updated.len()))
        {
            return false;
        }
        for node_idx in &tail {
            self.update_property_indices_for_add(node_type, *node_idx);
        }
        for mv in &moves {
            self.apply_update_move(node_type, mv);
        }
        if constrained {
            // Vacate before claiming, so a tuple one row gives up is free for
            // the row that takes it. `release_unique_claims` only removes a
            // tuple this node still occupies, so a release can never strip a
            // claim another row of the same batch just committed.
            for pre in updated {
                self.release_unique_claims(&pre.claims, pre.node_idx);
            }
            for node_idx in tail.iter().chain(updated.iter().map(|pre| &pre.node_idx)) {
                let claims = self.stored_unique_claims(node_type, *node_idx);
                self.commit_unique_claims(&claims, *node_idx);
            }
        }
        true
    }

    /// Whether claiming `rows` nodes' unique tuples one at a time is cheaper
    /// than re-deriving the type's occupancy from every member.
    ///
    /// The create arm's counterpart to [`Self::folding_moves_beats_a_rebuild`],
    /// and it matters for the *first* bulk load into a constrained type: every
    /// row is a create, so a per-row claim would pay a reader, a map and a
    /// tuple clone per node where `build_unique_index` pays one read. The
    /// weight is the ratio between those two per-node costs, and the members
    /// count already includes this batch's creates — so a first load, where
    /// rows *are* the members, always takes the rebuild it took before.
    fn claiming_beats_a_rebuild(&self, node_type: &str, rows: usize) -> bool {
        /// Work to claim one node's tuples against re-deriving one member's.
        const CLAIM_ELEMENT_WEIGHT: usize = 5;
        let members = self.type_indices.get(node_type).map_or(0, |m| m.len());
        rows.saturating_mul(CLAIM_ELEMENT_WEIGHT) <= members
    }

    /// Which updated rows actually have to move, and where.
    ///
    /// The comparison is against the value **as stored now**, not against what
    /// the row asked for: a `conflict_handling` mode that skipped or merged the
    /// write leaves the old value in place, and this then correctly plans
    /// nothing. That is what makes the fold independent of which conflict mode
    /// the batch ran under — and it is why the overwhelmingly common upsert (a
    /// row re-asserting the values it already had) costs a read per maintained
    /// property and no bucket edit at all.
    ///
    /// Read-only, deliberately: the count it returns is what
    /// [`Self::folding_moves_beats_a_rebuild`] decides on, and that decision
    /// has to be free to reverse.
    fn plan_update_moves(
        &mut self,
        node_type: &str,
        updated: &[UpdatedRowPreImage],
    ) -> Vec<UpdateMove> {
        let mut moves = Vec::new();
        for pre in updated {
            for (property, old) in &pre.indexed {
                let landed = self.indexed_value(node_type, property, pre.node_idx);
                match (old, landed) {
                    (None, None) => {}
                    (Some(before), Some(after)) if *before == after => {}
                    (old, landed) => moves.push(UpdateMove {
                        node_idx: pre.node_idx,
                        property: property.clone(),
                        old: old.clone(),
                        landed,
                    }),
                }
            }
        }
        moves
    }

    /// Apply one planned move: vacate the old bucket, join the new one.
    fn apply_update_move(&mut self, node_type: &str, mv: &UpdateMove) {
        match (&mv.old, &mv.landed) {
            (Some(before), None) => {
                // The value is gone (a `replace` that dropped the column).
                // Vacating without joining is what a rebuild does with it:
                // `create_index` files no bucket for an absent value.
                self.update_property_indices_for_remove(
                    node_type,
                    mv.node_idx,
                    &mv.property,
                    before,
                );
            }
            (old, Some(after)) => {
                self.update_property_indices_for_set(
                    node_type,
                    mv.node_idx,
                    &mv.property,
                    old.as_ref(),
                    after,
                );
            }
            (None, None) => {}
        }
    }

    /// Whether folding `moves` bucket moves is cheaper than rebuilding the
    /// type's covering indexes.
    ///
    /// **A move is not O(1).** Vacating a bucket is a `retain` over its
    /// members, and a composite move rescans the whole composite index to find
    /// the buckets the node currently sits in. So the fold is
    /// O(moves × bucket) against the rebuild's O(members) per index: far
    /// cheaper for the shape it exists for (ten rows into a 200k-row type) and
    /// *quadratic* for a bulk re-load that moves a large fraction of the type —
    /// which is the shape `refresh_indexes_for_type` was built for and handles
    /// in one linear pass.
    ///
    /// Both sides are counted in element visits, with the rebuild's visit
    /// weighted: reading a node's value, cloning it and hashing it into a map
    /// is an order of magnitude more work than comparing two `NodeIndex`es, and
    /// the weight keeps the comparison from declining a fold that is obviously
    /// cheaper. It is a cost *model*, not a tuned constant — it decides between
    /// two correct paths, so being wrong costs time and never an answer.
    fn folding_moves_beats_a_rebuild(&self, node_type: &str, moves: usize) -> bool {
        if moves == 0 {
            return true;
        }
        /// Work per rebuilt element (value read + clone + hash insert) against
        /// work per folded element (a `NodeIndex` compare).
        const REBUILD_ELEMENT_WEIGHT: usize = 16;

        let members = self.type_indices.get(node_type).map_or(0, |m| m.len());
        let mut covering = 0usize;
        // The longest bucket a single move may have to walk.
        let mut worst_bucket = 1usize;
        for ((nt, _), index) in &self.property_indices {
            if nt == node_type {
                covering += 1;
                worst_bucket = worst_bucket.max(members / index.len().max(1));
            }
        }
        for ((nt, _), index) in &self.range_indices {
            if nt == node_type {
                covering += 1;
                worst_bucket = worst_bucket.max(members / index.len().max(1));
            }
        }
        if self.composite_indices.keys().any(|(nt, _)| nt == node_type) {
            covering += self
                .composite_indices
                .keys()
                .filter(|(nt, _)| nt == node_type)
                .count();
            // A composite move scans every bucket of the index looking for the
            // node, so its walk is the whole index rather than one bucket.
            worst_bucket = worst_bucket.max(members);
        }
        if covering == 0 {
            // Constraint-only type: occupancy bookkeeping is hash work per row,
            // with no bucket to walk.
            return true;
        }
        moves.saturating_mul(worst_bucket)
            <= members
                .saturating_mul(covering)
                .saturating_mul(REBUILD_ELEMENT_WEIGHT)
    }

    /// The properties of `node_type` whose values a fold has to watch: every
    /// property carrying a single-value index, plus every member of a composite
    /// index on the type.
    ///
    /// A composite member need not carry an index of its own, and a change to
    /// it still moves the node within the composite index — reading only the
    /// single-value set would leave that node in a stale composite bucket.
    pub(crate) fn maintained_index_properties(&self, node_type: &str) -> Vec<String> {
        let mut properties: std::collections::BTreeSet<String> = self
            .single_value_indexed_properties(node_type)
            .into_iter()
            .collect();
        for (nt, members) in self.composite_indices.keys() {
            if nt == node_type {
                properties.extend(members.iter().cloned());
            }
        }
        properties.into_iter().collect()
    }

    /// Read everything a fold will need to *undo* about `node_idx` before the
    /// batch overwrites it: the old value of each maintained property, and the
    /// unique tuples the node currently occupies.
    ///
    /// Called once per updated node, before the batch executes — after it, the
    /// old values no longer exist anywhere. `properties` is
    /// [`Self::maintained_index_properties`], resolved once per call rather
    /// than per row.
    pub(crate) fn capture_update_pre_image(
        &mut self,
        node_type: &str,
        node_idx: NodeIndex,
        properties: &[String],
    ) -> UpdatedRowPreImage {
        let indexed = properties
            .iter()
            .map(|property| {
                let value = self.indexed_value(node_type, property, node_idx);
                (property.clone(), value)
            })
            .collect();
        let claims = if self.type_has_unique_constraints(node_type) {
            self.stored_unique_claims(node_type, node_idx)
        } else {
            Vec::new()
        };
        UpdatedRowPreImage {
            node_idx,
            indexed,
            claims,
        }
    }

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
        #[cfg(test)]
        TYPE_INDEX_REBUILDS.set(TYPE_INDEX_REBUILDS.get() + 1);
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

    /// The value an index *rebuild* would file `node_idx` under for the index
    /// registered as `property` — [`Self::create_index`]'s read, for one node.
    ///
    /// Incremental maintenance reads through here so it agrees with a rebuild
    /// **by construction**. Index keys carry the user's spelling, but their
    /// contents are the *resolved* field's values, so reading the node by the
    /// user-facing key files it under a value the matcher's scan can never
    /// produce — the phantom-row / poisoned-bucket divergence this replaced.
    fn indexed_value(
        &mut self,
        node_type: &str,
        property: &str,
        node_idx: NodeIndex,
    ) -> Option<Value> {
        let reader = self.property_reader(node_type, property);
        self.read_indexed(&reader, node_idx)
    }

    /// The properties carrying a single-value (hash or range) index on
    /// `node_type`, **deduplicated**: a property carrying both kinds appears
    /// once in each store, so iterating the two key sets naively appended the
    /// node twice per bucket and an indexed `MATCH` returned it twice.
    fn single_value_indexed_properties(&self, node_type: &str) -> Vec<String> {
        let mut properties: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        for (nt, property) in self
            .property_indices
            .keys()
            .chain(self.range_indices.keys())
        {
            if nt == node_type {
                properties.insert(property.as_str());
            }
        }
        properties.into_iter().map(str::to_string).collect()
    }

    /// The single-value index keys on `node_type` whose contents are drawn from
    /// `resolved` — the field a write to it moves the node within.
    ///
    /// Usually just `resolved` itself. It differs only for `id` / `title`,
    /// which an index may also be registered under the type's *alias* spelling
    /// for (`create_index("Term", "term_name")` on a title-aliased type builds
    /// its buckets from `get_node_title`), and those keys have to move with the
    /// field, not with the spelling of the statement that wrote it.
    fn index_keys_for_field(&self, node_type: &str, resolved: &str) -> Vec<String> {
        if self.id_field_aliases.is_empty() && self.title_field_aliases.is_empty() {
            return vec![resolved.to_string()];
        }
        let mut keys: Vec<String> = self
            .single_value_indexed_properties(node_type)
            .into_iter()
            .filter(|property| self.resolve_alias(node_type, property) == resolved)
            .collect();
        if !keys.iter().any(|property| property == resolved) {
            keys.push(resolved.to_string());
        }
        keys
    }

    /// Update property, composite, and range indices after a new node is added.
    /// Only updates indices that already exist for this node_type.
    pub fn update_property_indices_for_add(&mut self, node_type: &str, node_idx: NodeIndex) {
        if !self.type_has_user_indexes(node_type) {
            return;
        }
        note_maintenance_pass();
        for property in self.single_value_indexed_properties(node_type) {
            let Some(value) = self.indexed_value(node_type, &property, node_idx) else {
                continue;
            };
            let key = (node_type.to_string(), property);
            self.note_property_append(&key, &value, node_idx);
            if let Some(value_map) = self.property_indices.get_mut(&key) {
                value_map.entry_or_default(&value).push(node_idx);
            }
            self.note_range_append(&key, &value, node_idx);
            if let Some(btree) = self.range_indices.get_mut(&key) {
                btree.entry_or_default(&value).push(node_idx);
            }
        }

        let comp_keys: Vec<CompositeIndexKey> = self
            .composite_indices
            .keys()
            .filter(|(nt, _)| nt == node_type)
            .cloned()
            .collect();
        for key in comp_keys {
            let values: Vec<Value> = key
                .1
                .iter()
                .map(|property| {
                    self.indexed_value(node_type, property, node_idx)
                        .unwrap_or(Value::Null)
                })
                .collect();
            if values.iter().all(|v| matches!(v, Value::Null)) {
                continue;
            }
            let comp_val = CompositeValue(values);
            self.note_composite_append(&key, &comp_val, node_idx);
            if let Some(comp_map) = self.composite_indices.get_mut(&key) {
                comp_map.entry_or_default(&comp_val).push(node_idx);
            }
        }
    }

    /// Vacate `node_idx` from the `value` bucket of the hash and range indices
    /// keyed `key`, journalling each eviction first.
    ///
    /// Vacating the old bucket and joining the new one are journalled
    /// separately, and each capture has to read the map as it stands just
    /// before its own edit — hence the split borrows.
    fn evict_from_single_value_indexes(&mut self, key: &IndexKey, value: &Value, node: NodeIndex) {
        self.note_property_eviction(key, value, node);
        if let Some(value_map) = self.property_indices.get_mut(key) {
            if let Some(indices) = value_map.get_mut(value) {
                indices.retain(|&idx| idx != node);
                if indices.is_empty() {
                    value_map.remove(value);
                }
            }
        }
        self.note_range_eviction(key, value, node);
        if let Some(btree) = self.range_indices.get_mut(key) {
            if let Some(indices) = btree.get_mut(value) {
                indices.retain(|&idx| idx != node);
                if indices.is_empty() {
                    btree.remove(value);
                }
            }
        }
    }

    /// Whether `node_type` carries any of the three index families incremental
    /// maintenance edits — hash equality, range, composite.
    ///
    /// The gate on every incremental updater. A graph with no indexes at all
    /// answers in three `is_empty` checks; one with indexes on *other* types
    /// pays a scan of the (small) key sets, which is still per write rather
    /// than per index bucket.
    ///
    /// Unique constraints are deliberately **not** consulted: their occupancy
    /// lives in `unique_indices`, is planned before the write and redeemed
    /// after it (`constraints::plan_property_write` /
    /// `apply_property_write_plan`), and is never touched by the updaters this
    /// gates. Disk-backed persistent `PropertyIndex` stores are out of scope
    /// for the same reason as in [`Self::refresh_indexes_for_type`] — they
    /// never land in `property_indices`, so the updaters cannot see them
    /// either.
    pub(crate) fn type_has_user_indexes(&self, node_type: &str) -> bool {
        if self.property_indices.is_empty()
            && self.range_indices.is_empty()
            && self.composite_indices.is_empty()
        {
            return false;
        }
        self.property_indices
            .keys()
            .chain(self.range_indices.keys())
            .any(|(nt, _)| nt == node_type)
            || self.composite_indices.keys().any(|(nt, _)| nt == node_type)
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
        // Nothing to move the node between: no bucket edit, and therefore no
        // value read-back, no resolved-field `String`, no key set to build.
        if !self.type_has_user_indexes(node_type) {
            return;
        }
        note_maintenance_pass();
        // The value that actually landed, read the way a rebuild reads it, so
        // storage that keeps the authoritative copy elsewhere (a columnar
        // master, the node's id/title fields) buckets the node under what a
        // `MATCH` will find. Falls back to the caller's value for a backend
        // whose granular read cannot see the write yet.
        let landed = self.indexed_value(node_type, property, node_idx);
        let landed = landed.unwrap_or_else(|| new_value.clone());
        let field = self.resolve_alias(node_type, property).to_string();
        for indexed_as in self.index_keys_for_field(node_type, &field) {
            let key = (node_type.to_string(), indexed_as);
            if let Some(old_val) = old_value {
                self.evict_from_single_value_indexes(&key, old_val, node_idx);
            }
            self.note_property_append(&key, &landed, node_idx);
            if let Some(value_map) = self.property_indices.get_mut(&key) {
                value_map.entry_or_default(&landed).push(node_idx);
            }
            self.note_range_append(&key, &landed, node_idx);
            if let Some(btree) = self.range_indices.get_mut(&key) {
                btree.entry_or_default(&landed).push(node_idx);
            }
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
        if !self.type_has_user_indexes(node_type) {
            return;
        }
        note_maintenance_pass();
        let field = self.resolve_alias(node_type, property).to_string();
        for indexed_as in self.index_keys_for_field(node_type, &field) {
            let key = (node_type.to_string(), indexed_as);
            self.evict_from_single_value_indexes(&key, old_value, node_idx);
        }

        // Update any composite indices that include this property
        self.update_composite_indices_for_property_change(node_type, node_idx, property);
    }

    /// Re-index a single node in all composite indices that include the changed property.
    /// Reads current node properties to build the new composite value.
    ///
    /// Membership is decided by *field*, not spelling: a composite index
    /// registered under a type's title-alias spelling holds titles, so a write
    /// to `title` moves the node within it (same reasoning as
    /// [`Self::index_keys_for_field`]).
    fn update_composite_indices_for_property_change(
        &mut self,
        node_type: &str,
        node_idx: NodeIndex,
        changed_property: &str,
    ) {
        let changed_field = self.resolve_alias(node_type, changed_property);
        let comp_keys: Vec<CompositeIndexKey> = self
            .composite_indices
            .keys()
            .filter(|(nt, props)| {
                nt == node_type
                    && props
                        .iter()
                        .any(|p| self.resolve_alias(node_type, p) == changed_field)
            })
            .cloned()
            .collect();

        if comp_keys.is_empty() {
            return;
        }

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

            // Build the new composite value the way `create_composite_index`
            // builds every other one — see `indexed_value`.
            let new_values: Vec<Value> = key
                .1
                .iter()
                .map(|p| {
                    self.indexed_value(node_type, p, node_idx)
                        .unwrap_or(Value::Null)
                })
                .collect();
            if new_values.iter().any(|v| !matches!(v, Value::Null)) {
                let value = CompositeValue(new_values);
                // After the evictions, so `bucket_was_new` reflects the map
                // the append actually lands in.
                self.note_composite_append(&key, &value, node_idx);
                if let Some(comp_map) = self.composite_indices.get_mut(&key) {
                    comp_map.entry_or_default(&value).push(node_idx);
                }
            }
        }
    }
}
