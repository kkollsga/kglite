//! The columnar-store rebuild pipeline on `DirGraph` — `enable_columnar` (the
//! internal consolidation primitive save/vacuum/unspill/enable_disk_mode all
//! funnel through) and its two private halves, which build the fresh per-type
//! stores and then re-point every node at its new row. Split out of `mod.rs`
//! to keep it under the god-file LoC ceiling; these three are one pass and
//! have no other callers.

use std::collections::HashMap;
use std::sync::Arc;

use petgraph::graph::NodeIndex;
use rustc_hash::FxHashMap;

#[cfg(test)]
use super::note_columnar_rebuild;
use super::DirGraph;
use crate::datatypes::values::Value;
use crate::graph::schema::{ColumnarRow, InternedKey, PropertyStorage};
use crate::graph::storage::column_store::ColumnStore;
use crate::graph::storage::{GraphRead, GraphWrite};

impl DirGraph {
    /// Consolidate every node's properties into its type's `ColumnStore`.
    ///
    /// **This is the internal consolidation primitive, not a mode switch.**
    /// Properties are columnar from construction (`dir_graph::node_write`,
    /// `mutation::batch`), so on a settled graph this call has nothing to do
    /// and says so in O(N) — the fast path below. What it still *does* is
    /// rebuild: it is the one pass that reclaims rows deleted nodes left
    /// behind, restores ascending row order after a create reused a freed
    /// petgraph slot, and re-derives every column's type from the type's
    /// metadata. Its callers are `save()` (where those three are the
    /// difference between a lean, correctly-bound file and a corrupt one),
    /// `vacuum`, `unspill`, and `enable_disk_mode`.
    ///
    /// It is `pub(crate)`: there is no regime for a caller to enable, so the
    /// only way in from outside the crate is the operation that needs the
    /// pass done — [`crate::api::io::prepare_kgl_write`].
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
    /// This guard once compared each node's own store `Arc` with the graph's
    /// by pointer, because `PropertyStorage::insert` on a columnar
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
    pub(crate) fn enable_columnar(&mut self) {
        if self.columnar_rebuild_needed() {
            self.rebuild_column_stores();
        } else {
            // The stores are already the shape a save wants, but the
            // *memory limit* still has to be honoured: this is the call
            // that spills, and taking the fast path used to skip the
            // spill with it. Harmless while a fresh graph always rebuilt
            // here; a silent contract break once construction is columnar
            // and the fast path is the normal case
            // (`test_spill_when_over_limit`).
            self.maybe_spill_columns();
        }
    }

    /// Whether [`Self::enable_columnar`] would rebuild the stores: the read-only
    /// half of its decision, so a writer that must not touch a shared graph
    /// (an online backup of a published snapshot) can ask first and rebuild a
    /// private copy only when the answer is yes.
    pub(crate) fn columnar_rebuild_needed(&self) -> bool {
        if self.column_store_count() > 0 {
            let backend = &self.graph;
            // Rows a type has handed out so far, walking nodes in ascending
            // index order — the order the file is written and read back in.
            // FxHash, not the std SipHasher: `InternedKey` is already a
            // well-distributed FNV `u64` and this probes once per node of the
            // graph. `RandomState` was the top self symbol (14.9%) of a
            // 1M-node save profile.
            let mut next_row: FxHashMap<InternedKey, u32> = FxHashMap::default();
            let any_drift = self
                .graph
                .node_indices()
                .filter_map(|idx| self.graph.node_weight(idx))
                .any(|n| match &n.properties {
                    PropertyStorage::Columnar(row) => {
                        let row_id = row.row_id();
                        // **Row order is part of the saved format.** The `.kgl`
                        // column section carries rows positionally, and the load
                        // path binds row k of a type to that type's k-th node in
                        // ascending node-index order — so a store whose rows are
                        // not in that order serializes every row against the
                        // wrong node (ids, titles and properties all shift, and
                        // the edges then appear to connect different nodes:
                        // petekSuite bug 4). `rebuild_column_stores` sorts by
                        // node index, which is what makes save-order equal
                        // load-order; this check is what decides the rebuild is
                        // needed. Rows are appended in creation order, and a
                        // creation reuses a freed petgraph slot, so any
                        // delete-then-create pair can put them out of order.
                        let expected = next_row.entry(n.node_type).or_insert(0);
                        let out_of_order = row_id != *expected;
                        *expected += 1;
                        out_of_order
                            || match backend.column_store(n.node_type) {
                                Some(graph_store) => {
                                    // A title written onto the inline
                                    // `node.title` rather than through the
                                    // store's `__title__` column. Every
                                    // in-engine title write goes through the
                                    // store now (`GraphWrite::set_node_title`),
                                    // so this fires only for the fallback that
                                    // takes when a type has no store to write
                                    // through — but it is what consolidates
                                    // that node's title, so it stays
                                    // (petekSuite bug 2). A consolidated/loaded
                                    // node has `node.title == Null`, so no
                                    // false drift.
                                    !matches!(n.title_ref(), Value::Null)
                                        && graph_store.get_title(row_id).as_ref()
                                            != Some(n.title_ref())
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
            return any_drift || orphaned_rows;
        }
        true
    }

    /// The O(N) half of [`DirGraph::enable_columnar`]: rebuild every type's
    /// store from live nodes and re-point the nodes at their rows.
    ///
    /// Split out so the idempotence guard above reads as the decision it is,
    /// and so the rebuild has a single entry point to count.
    pub(super) fn rebuild_column_stores(&mut self) {
        #[cfg(test)]
        note_columnar_rebuild();
        {}
        use crate::graph::storage::column_store::ColumnStore;

        // The store constructors read the type's shared schema, so derive it
        // first for a graph that has none (a fixture built straight through
        // `GraphWrite::add_node`, or a pre-metadata file).
        if self.type_schemas.is_empty() {
            self.rebuild_type_schemas();
        }
        // A type whose nodes were staged with inline properties (the
        // N-Triples and RDF loaders, `embedding_carry`) has no schema entry
        // once any other typed node exists, and the pass below skips a type
        // without one.
        self.register_staged_properties();

        // Build a ColumnStore per node type
        let mut stores: HashMap<String, ColumnStore> = HashMap::new();
        // Which node each row belongs to, per type — `row_owners[t][row_id]`.
        //
        // A `HashMap<NodeIndex, u32>` per type, which is what this was, records
        // exactly the same thing at one SipHash insert per node of the graph on
        // the way in and one per node on the way out. Rows are handed out
        // densely from 0 in the order this loop walks them, so the position in
        // a vector *is* the row id.
        let mut row_owners: HashMap<String, Vec<NodeIndex>> = HashMap::new();

        // Clean type_indices: remove entries for deleted/tombstoned nodes.
        {
            let graph_ref = &self.graph;
            self.type_indices
                .retain_all(|idx| graph_ref.node_weight(*idx).is_some());
        }

        // One buffer for every row of every type: the per-node
        // `Vec<(InternedKey, Value)>` this loop reads out of the old store was
        // allocated and freed once per node of the graph.
        let mut pairs: Vec<(InternedKey, Value)> = Vec::new();
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
            let mut type_row_owners: Vec<NodeIndex> = Vec::with_capacity(indices.len());

            // Build column rows in ascending node-index order so the saved row
            // order matches the load-side re-point, which enumerates
            // `type_indices` rebuilt in ascending node-index order (see
            // io/file.rs "Re-point nodes to columnar storage" +
            // rebuild_type_indices_and_schemas scanning node_indices()). This
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
                    // Push id/title for every node. For Columnar nodes, try the
                    // old column store first and fall back to the inline fields;
                    // for a staged `Map` node the inline fields are all there is.
                    let old_row = node.properties.columnar_row_id();
                    let old_store = old_row.and(self.graph.column_store(node.node_type));
                    let id_val = if let (Some(old_store), Some(old_row)) = (old_store, old_row) {
                        old_store
                            .get_id(old_row)
                            .unwrap_or_else(|| node.id_ref().clone())
                    } else {
                        node.id_ref().clone()
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
                        if !matches!(node.title_ref(), Value::Null) {
                            node.title_ref().clone()
                        } else {
                            old_store
                                .get_title(old_row)
                                .unwrap_or_else(|| node.title_ref().clone())
                        }
                    } else {
                        node.title_ref().clone()
                    };

                    store.push_id(&id_val);
                    store.push_title(&title_val);

                    // Collect properties from current storage
                    match &node.properties {
                        PropertyStorage::Map(map) => {
                            pairs.clear();
                            pairs.extend(map.iter().map(|(&k, v)| (k, v.clone())));
                        }
                        PropertyStorage::Columnar(row) => {
                            match self.graph.column_store(node.node_type) {
                                Some(store) => store.row_properties_into(row.row_id(), &mut pairs),
                                None => pairs.clear(),
                            }
                        }
                    }

                    let row_id = store.push_row(&pairs);
                    debug_assert_eq!(
                        row_id as usize,
                        type_row_owners.len(),
                        "rows must be handed out densely from 0 for the position \
                         in `row_owners` to be the row id"
                    );
                    type_row_owners.push(idx);
                }
            }

            stores.insert(node_type.to_string(), store);
            row_owners.insert(node_type.to_string(), type_row_owners);
        }

        self.install_rebuilt_column_stores(stores, &row_owners);
        // Spill to disk if over the memory limit. Through the one spill
        // routine, on the installed stores — this used to be a second copy of
        // `maybe_spill_columns` operating on the local map, which meant the
        // rebuild path and the standalone path could drift on which directory
        // they used and which store they picked first.
        self.maybe_spill_columns();
    }

    /// Register the property names and value types of every node still holding
    /// inline (`Map`) properties into its type's metadata and schema, so the
    /// rebuild gives those properties a column.
    ///
    /// One entry per distinct (type, property, value type), so a graph of a
    /// million alike nodes registers a handful of entries; a type that already
    /// carries a property costs one lookup.
    fn register_staged_properties(&mut self) {
        let mut seen: rustc_hash::FxHashSet<(InternedKey, InternedKey, &'static str)> =
            rustc_hash::FxHashSet::default();
        let mut found: Vec<(InternedKey, InternedKey, Value)> = Vec::new();
        {
            for idx in self.graph.node_indices() {
                let Some(node) = self.graph.node_weight(idx) else {
                    continue;
                };
                if let PropertyStorage::Map(map) = &node.properties {
                    for (&key, value) in map.iter() {
                        if seen.insert((node.node_type, key, value.type_name())) {
                            found.push((node.node_type, key, value.clone()));
                        }
                    }
                }
            }
        }
        for (node_type, key, value) in found {
            let type_name = self.interner.resolve(node_type).to_string();
            let key_name = self.interner.resolve(key).to_string();
            self.register_property_types(&type_name, &HashMap::from([(key_name, value)]));
            self.ensure_type_schema_keys(&type_name, &[key]);
        }
    }

    /// Second pass of the rebuild: publish the stores onto the backend and
    /// point each node at its row.
    fn install_rebuilt_column_stores(
        &mut self,
        stores: HashMap<String, ColumnStore>,
        row_owners: &HashMap<String, Vec<NodeIndex>>,
    ) {
        let arc_stores: HashMap<String, Arc<ColumnStore>> =
            stores.into_iter().map(|(t, s)| (t, Arc::new(s))).collect();

        for (node_type, type_row_owners) in row_owners {
            if arc_stores.contains_key(node_type) {
                for (row_id, &idx) in type_row_owners.iter().enumerate() {
                    let row_id = row_id as u32;
                    // Silent: re-pointing a node at its renumbered row changes no
                    // logical state, and a recording backend would otherwise log
                    // every live node as an update.
                    if let Some(node) = self.graph.node_weight_mut_silent(idx) {
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
                        node.clear_ident();
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

    /// Reclaim the column rows deleted nodes left behind, without moving any
    /// node: the commit-time counterpart of [`Self::check_auto_vacuum`] for
    /// bindings that publish whole graphs (`Session`).
    ///
    /// A delete only tombstones a node's row, and rows are append-only, so a
    /// create/delete steady state grew every type's store without bound while
    /// the live node count stayed flat. This rebuilds the stores once the dead
    /// rows pass the same floor (more than 100) and ratio
    /// ([`Self::auto_vacuum_threshold`]) the vacuum trigger uses, so retention
    /// is bounded by a fraction of the live rows and the O(live) rebuild is
    /// amortised over the deletes that earned it. `None` disables it, as it
    /// does the vacuum trigger.
    ///
    /// Unlike a vacuum no `NodeIndex` changes, so text/vector indexes, secondary
    /// labels and selections stay valid. Skipped where a rebuild would be
    /// wrong or wasteful: the mapped and disk backends, whose stores are
    /// file-backed on purpose. A copy-on-write overlay is collapsed first,
    /// because the rebuild re-points every live node at its renumbered row and
    /// would otherwise copy each one into the overlay; the collapse is O(graph)
    /// and runs only once the dead rows pass the floor and ratio above, so it
    /// is amortised over the deletes that earned it.
    ///
    /// Must run outside a statement window: it renumbers rows, which an open
    /// undo journal still names.
    pub(crate) fn compact_columns_if_fragmented(&mut self) -> bool {
        let Some(threshold) = self.auto_vacuum_threshold else {
            return false;
        };
        if self.graph.is_mapped() || self.graph.is_disk() {
            return false;
        }
        let (total, live) = self.columnar_row_census();
        let dead = total.saturating_sub(live);
        if dead <= 100 || (dead as f64 / total as f64) <= threshold {
            return false;
        }
        self.graph.flatten_fork();
        self.rebuild_columns_to_heap();
        true
    }
}
