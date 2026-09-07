//! Persistent property indexes for `DiskGraph`: build, lookup (eq +
//! prefix), and global (cross-type) variants.
//!
//! Split out of `graph.rs` to keep that file under the 2,500-line cap.
//! Lives in a sibling `impl DiskGraph {}` block.

use crate::datatypes::values::Value;
use crate::graph::schema::InternedKey;
use petgraph::graph::NodeIndex;
use std::collections::HashMap;
use std::sync::Arc;

use super::graph::DiskGraph;
use super::property_index;

#[cfg(test)]
thread_local! {
    static BUILD_FAILPOINT: std::cell::Cell<Option<&'static str>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) struct PropertyIndexBuildFailpoint;

#[cfg(test)]
impl Drop for PropertyIndexBuildFailpoint {
    fn drop(&mut self) {
        BUILD_FAILPOINT.with(|point| point.set(None));
    }
}

#[cfg(test)]
pub(crate) fn fail_property_index_build(stage: &'static str) -> PropertyIndexBuildFailpoint {
    BUILD_FAILPOINT.with(|point| point.set(Some(stage)));
    PropertyIndexBuildFailpoint
}

fn property_index_build_failpoint(stage: &'static str) -> std::io::Result<()> {
    #[cfg(test)]
    if BUILD_FAILPOINT.with(|point| point.get() == Some(stage)) {
        return Err(std::io::Error::other(format!(
            "injected {stage} property-index build failure"
        )));
    }
    let _ = stage;
    Ok(())
}

impl DiskGraph {
    /// Build (or rebuild) a persistent string property index for
    /// `(node_type, property)`. Writes four files to `data_dir` and
    /// caches the handle. Subsequent `lookup_property_eq` calls use the
    /// index; the planner sees it via the `GraphRead::lookup_by_property_eq`
    /// trait method.
    ///
    /// Only `TypedColumn::Str` columns are indexable today — the property
    /// must exist on the type's ColumnStore as a string column. Non-string
    /// or missing properties are a no-op that returns `Ok(())`; the index
    /// will simply contain zero entries and all lookups will miss.
    pub fn build_property_index(
        &mut self,
        node_type: &str,
        property: &str,
    ) -> std::io::Result<usize> {
        self.prepare_mutation()?;
        let index_key = (node_type.to_string(), property.to_string());
        let type_key = InternedKey::from_str(node_type);
        let type_u64 = type_key.as_u64();
        let prop_key = InternedKey::from_str(property);

        // Three ways to resolve a property to a string value per node:
        //   1. Title/id alias columns (checked via helpers below) — covers
        //      `label`, `nid`, and any user-chosen title/id field names.
        //   2. Regular schema column via `get_str_by_slot`.
        //   3. Fall back to NodeData::get_property, which is the arena
        //      path used by the pattern matcher — slower but correct for
        //      exotic cases (non-columnar properties, map storage).
        let col_store = self.column_stores.get(&type_key);
        let schema_slot = col_store.and_then(|cs| cs.schema().slot(prop_key));
        // Heuristic: "title" or "id" literals, and anything stored outside
        // the regular schema, goes through the NodeData materialisation
        // path so title/id aliases and mapped-mode stores resolve
        // correctly. Everything else reads directly from the column.
        let use_slot_path = schema_slot.is_some();

        let node_bound = self.node_slot_len();
        let mut entries: Vec<(String, u32)> = Vec::with_capacity(node_bound);
        for i in 0..node_bound {
            let nslot = self.node_slot(i);
            if !nslot.is_alive() || nslot.node_type != type_u64 {
                continue;
            }
            // Try paths in order of specificity:
            //   1. Regular schema column (`get_str_by_slot`) — fast path.
            //   2. Title column (`get_title`) — covers `label`/`name`/
            //      any user-chosen title alias.
            //   3. Id column (`get_id`) — covers `nid` and other id
            //      aliases when the user explicitly indexes the id.
            let maybe_str: Option<String> = if use_slot_path {
                col_store
                    .and_then(|cs| cs.get_str_by_slot(nslot.row_id, schema_slot.unwrap()))
                    .map(str::to_string)
            } else if let Some(cs) = col_store {
                // Not in schema — try title, then id. If both return a
                // non-empty String, prefer title (which is what users
                // typically mean when aliasing `label` / `name` / ...).
                let from_title = cs.get_title(nslot.row_id).and_then(|v| match v {
                    Value::String(s) if !s.is_empty() => Some(s),
                    _ => None,
                });
                if from_title.is_some() {
                    from_title
                } else {
                    cs.get_id(nslot.row_id).and_then(|v| match v {
                        Value::String(s) if !s.is_empty() => Some(s),
                        _ => None,
                    })
                }
            } else {
                None
            };
            if let Some(s) = maybe_str {
                entries.push((s, i as u32));
            }
        }

        let count = entries.len();
        // Evict the cached index before rebuilding it. `PropertyIndex::build`
        // truncates the same `keys`/`offsets`/`ids` files that a cached entry
        // still has memory-mapped, and Windows refuses to re-create a mapped
        // file (`ERROR_USER_MAPPED_FILE`). Removing the key (rather than
        // storing `None`, which means "no such index") lets a concurrent
        // lookup fall back to opening the bundle from disk. A legacy-value
        // mask remains authoritative until the replacement is published.
        self.property_indexes.write().unwrap().remove(&index_key);
        property_index_build_failpoint("typed")?;
        let idx = property_index::PropertyIndex::build(
            self.active_write_dir(),
            node_type,
            property,
            entries,
        )?;
        self.property_indexes
            .write()
            .unwrap()
            .insert(index_key.clone(), Some(Arc::new(idx)));
        self.index_freshness
            .mark_typed_built(index_key.clone(), node_bound as u32);
        self.removed_property_indexes.remove(&index_key);
        self.legacy_invalidated_property_indexes.remove(&index_key);
        Ok(count)
    }

    /// Drop a typed persistent index in the writer overlay. The selected
    /// generation remains immutable; the next save omits its bundle.
    pub fn drop_property_index(
        &mut self,
        node_type: &str,
        property: &str,
    ) -> std::io::Result<bool> {
        self.prepare_mutation()?;
        let existed = self.has_property_index(node_type, property);
        if !existed {
            return Ok(false);
        }
        self.removed_property_indexes
            .insert((node_type.to_string(), property.to_string()));
        self.property_indexes
            .write()
            .unwrap()
            .insert((node_type.to_string(), property.to_string()), None);
        self.index_freshness
            .forget_typed(&(node_type.to_string(), property.to_string()));
        property_index::PropertyIndex::remove_files(self.active_write_dir(), node_type, property)?;
        Ok(true)
    }

    /// The typed bundle serving `(node_type, property)`, or `None` when there
    /// is none — cache first, then the filesystem, caching whichever answer it
    /// finds so a repeat miss does not stat again.
    ///
    /// Says nothing about freshness: [`Self::serving_property_index`] is what a
    /// lookup asks.
    fn cached_property_index(
        &self,
        key: &(String, String),
    ) -> Option<Arc<property_index::PropertyIndex>> {
        {
            let read = self.property_indexes.read().unwrap();
            if let Some(slot) = read.get(key) {
                return slot.clone();
            }
        }
        let opened = property_index::PropertyIndex::open(&self.data_dir, &key.0, &key.1)
            .ok()
            .flatten()
            .map(Arc::new);
        self.property_indexes
            .write()
            .unwrap()
            .insert(key.clone(), opened.clone());
        opened
    }

    /// The typed bundle for `(node_type, property)`, **only if it still covers
    /// the graph**.
    ///
    /// `None` means *unknown* — go scan — and covers three cases a caller must
    /// not distinguish: no bundle, a masked legacy bundle, and a bundle the
    /// graph has moved under. Nothing maintains an mmap bundle, so the third is
    /// as unanswerable as the first: returning `Some(hits)` from it reports
    /// "no such row" for every row written since the build (deep-scan item 3).
    fn serving_property_index(
        &self,
        node_type: &str,
        property: &str,
    ) -> Option<Arc<property_index::PropertyIndex>> {
        let key = (node_type.to_string(), property.to_string());
        if self.legacy_invalidated_property_indexes.contains(&key) {
            return None;
        }
        let index = self.cached_property_index(&key)?;
        self.index_freshness
            .typed_is_fresh(&key, self.node_slot_len() as u32)
            .then_some(index)
    }

    /// Exact-match lookup. Returns `None` when no index can answer for
    /// `(node_type, property)` — none built, or one the graph has moved under
    /// since it was — and `Some(Vec)` (possibly empty) from an index that
    /// provably covers every live row. The planner uses the distinction to
    /// decide whether to route through the fast path or fall back to scan.
    pub fn lookup_property_eq(
        &self,
        node_type: &str,
        property: &str,
        value: &str,
    ) -> Option<Vec<NodeIndex>> {
        Some(
            self.serving_property_index(node_type, property)?
                .lookup_eq_str(value),
        )
    }

    /// Prefix lookup (STARTS WITH). Same `None`/`Some` semantics as
    /// [`lookup_property_eq`].
    pub fn lookup_property_prefix(
        &self,
        node_type: &str,
        property: &str,
        prefix: &str,
        limit: usize,
    ) -> Option<Vec<NodeIndex>> {
        Some(
            self.serving_property_index(node_type, property)?
                .lookup_prefix_str(prefix, limit),
        )
    }

    /// Whether an index has been built for `(node_type, property)`.
    /// Checks the cache first, then the filesystem.
    pub fn has_property_index(&self, node_type: &str, property: &str) -> bool {
        let key = (node_type.to_string(), property.to_string());
        if self.legacy_invalidated_property_indexes.contains(&key) {
            return false;
        }
        if let Some(slot) = self.property_indexes.read().unwrap().get(&key) {
            return slot.is_some();
        }
        property_index::PropertyIndex::open(&self.data_dir, node_type, property)
            .ok()
            .flatten()
            .is_some()
    }

    /// Build a cross-type global index for `property`. Scans every
    /// alive `DiskNodeSlot` and emits one `(string_value, NodeIndex)`
    /// entry per node where `property` resolves to a non-empty string
    /// (regular column, title alias, or id alias — same resolution
    /// order as [`build_property_index`]).
    ///
    /// Powers untyped patterns like `MATCH (n {label: 'X'})` and the
    /// `search(text)` helper. Re-run whenever the graph is rebuilt.
    pub fn build_global_property_index(&mut self, property: &str) -> std::io::Result<usize> {
        self.prepare_mutation()?;
        let prop_key = InternedKey::from_str(property);
        let node_bound = self.node_slot_len();
        let mut entries: Vec<(String, u32)> = Vec::with_capacity(node_bound / 2);

        // Cache per-type (column_store, schema_slot) lookups so every
        // node in the same type reuses the slot resolution.
        type ColStore = Arc<crate::graph::storage::column_store::ColumnStore>;
        type TypeCacheEntry = Option<(ColStore, Option<u16>)>;
        let mut type_cache: HashMap<u64, TypeCacheEntry> = HashMap::new();

        for i in 0..node_bound {
            let nslot = self.node_slot(i);
            if !nslot.is_alive() {
                continue;
            }
            let cached = type_cache.entry(nslot.node_type).or_insert_with(|| {
                let tk = InternedKey::from_u64(nslot.node_type);
                self.column_stores.get(&tk).cloned().map(|cs| {
                    let slot = cs.schema().slot(prop_key);
                    (cs, slot)
                })
            });
            let Some((col_store, schema_slot)) = cached else {
                continue;
            };
            let maybe_str: Option<String> = if let Some(slot) = schema_slot {
                col_store
                    .get_str_by_slot(nslot.row_id, *slot)
                    .map(str::to_string)
            } else {
                let from_title = col_store.get_title(nslot.row_id).and_then(|v| match v {
                    Value::String(s) if !s.is_empty() => Some(s),
                    _ => None,
                });
                from_title.or_else(|| {
                    col_store.get_id(nslot.row_id).and_then(|v| match v {
                        Value::String(s) if !s.is_empty() => Some(s),
                        _ => None,
                    })
                })
            };
            if let Some(s) = maybe_str {
                if !s.is_empty() {
                    entries.push((s, i as u32));
                }
            }
        }

        let count = entries.len();
        // Same rebuild-over-a-live-mapping hazard as `build_property_index`:
        // release the cached bundle before `build_global` truncates the files
        // it maps. `save_disk` rebuilds the `title` and `nid` global indexes on
        // every save, so on Windows the second save of a graph would otherwise
        // fail here. A legacy-value mask remains authoritative until the
        // replacement is published.
        self.global_indexes.write().unwrap().remove(property);
        property_index_build_failpoint("global")?;
        let idx = property_index::PropertyIndex::build_global(
            self.active_write_dir(),
            property,
            entries,
        )?;
        self.global_indexes
            .write()
            .unwrap()
            .insert(property.to_string(), Some(Arc::new(idx)));
        self.index_freshness
            .mark_global_built(property, node_bound as u32);
        self.legacy_invalidated_global_indexes.remove(property);
        Ok(count)
    }

    /// The global bundle for `property`, cache first then filesystem.
    /// Freshness is [`Self::serving_global_index`]'s question.
    fn cached_global_index(&self, property: &str) -> Option<Arc<property_index::PropertyIndex>> {
        {
            let read = self.global_indexes.read().unwrap();
            if let Some(slot) = read.get(property) {
                return slot.clone();
            }
        }
        let opened = property_index::PropertyIndex::open_global(&self.data_dir, property)
            .ok()
            .flatten()
            .map(Arc::new);
        self.global_indexes
            .write()
            .unwrap()
            .insert(property.to_string(), opened.clone());
        opened
    }

    /// The global bundle for `property`, only if it still covers the graph.
    ///
    /// Every disk `save()` auto-builds the `title` and `nid` globals, so this
    /// gate is what a graph gets for free: without it, a save+load armed a
    /// bundle that answered "no such node" for everything ingested since the
    /// load (deep-scan item 2).
    fn serving_global_index(&self, property: &str) -> Option<Arc<property_index::PropertyIndex>> {
        if self.legacy_invalidated_global_indexes.contains(property) {
            return None;
        }
        let index = self.cached_global_index(property)?;
        self.index_freshness
            .global_is_fresh(property, self.node_slot_len() as u32)
            .then_some(index)
    }

    /// Exact-match lookup across every node type for a cross-type
    /// global index. Same `None` = *unknown* contract as
    /// [`lookup_property_eq`].
    pub fn lookup_global_eq(&self, property: &str, value: &str) -> Option<Vec<NodeIndex>> {
        Some(self.serving_global_index(property)?.lookup_eq_str(value))
    }

    /// Prefix lookup (STARTS WITH) against the cross-type global
    /// index. Same `None`/`Some` semantics as [`lookup_global_eq`].
    pub fn lookup_global_prefix(
        &self,
        property: &str,
        prefix: &str,
        limit: usize,
    ) -> Option<Vec<NodeIndex>> {
        Some(
            self.serving_global_index(property)?
                .lookup_prefix_str(prefix, limit),
        )
    }

    /// Whether this graph has any persistent bundle whose freshness has to be
    /// tracked — the whole cost the write path pays when it does not.
    #[inline]
    pub(crate) fn tracks_index_freshness(&self) -> bool {
        self.index_freshness.tracks_anything(&self.data_dir)
    }

    /// A node of `node_type` was created at `slot`.
    #[inline]
    pub(crate) fn note_index_node_created(&self, slot: u32, node_type: &str) {
        self.index_freshness.note_created(slot, node_type);
    }

    /// A property of the node at `slot` was written; `None` for a caller that
    /// did not resolve the node's type.
    #[inline]
    pub(crate) fn note_index_property_written(&self, slot: u32, node_type: Option<&str>) {
        self.index_freshness.note_property_written(slot, node_type);
    }

    /// The node at `slot` was removed.
    #[inline]
    pub(crate) fn note_index_node_removed(&self, slot: u32) {
        self.index_freshness.note_removed(slot);
    }

    /// Whether the typed bundle for `(node_type, property)` is currently
    /// serving lookups — it exists *and* still covers the graph.
    ///
    /// Distinct from [`Self::has_property_index`], which answers "was one
    /// built?". Introspection needs both: `DROP INDEX` acts on existence,
    /// while a `describe()` hint that names an index a query will not use is
    /// an agent-facing claim the engine contradicts.
    pub(crate) fn property_index_is_serving(&self, node_type: &str, property: &str) -> bool {
        self.serving_property_index(node_type, property).is_some()
    }

    /// Whether a global bundle for `property` exists but is refusing to
    /// answer because the graph has moved under it.
    ///
    /// The distinction a plain `None` from [`Self::lookup_global_eq`] cannot
    /// carry: "no index here, that answer is as good as it gets" versus "there
    /// is an index and it cannot be trusted". Only the second is worth a scan.
    pub(crate) fn global_index_is_declining(&self, property: &str) -> bool {
        if self.legacy_invalidated_global_indexes.contains(property) {
            return false;
        }
        self.cached_global_index(property).is_some()
            && !self
                .index_freshness
                .global_is_fresh(property, self.node_slot_len() as u32)
    }

    /// Every persistent bundle reachable from this graph, as
    /// `(typed pairs, global properties)`.
    ///
    /// Unions the published generation with the writer workspaces, exactly the
    /// set [`Self::copy_persisted_indexes`] would carry into the next
    /// generation, plus whatever the caches have opened. Legacy-named bundles
    /// are invisible to the scanners by design (their filenames destroyed the
    /// identity), so a rebuild cannot reach them — they stay masked by the
    /// freshness gate instead, which is the correct answer for a bundle whose
    /// key nothing can reconstruct.
    pub(crate) fn persisted_index_names(&self) -> (Vec<(String, String)>, Vec<String>) {
        let mut typed: std::collections::BTreeSet<(String, String)> = self
            .property_indexes
            .read()
            .unwrap()
            .iter()
            .filter(|(_, slot)| slot.is_some())
            .map(|(key, _)| key.clone())
            .collect();
        let mut global: std::collections::BTreeSet<String> = self
            .global_indexes
            .read()
            .unwrap()
            .iter()
            .filter(|(_, slot)| slot.is_some())
            .map(|(property, _)| property.clone())
            .collect();
        let mut dirs = vec![self.data_dir.clone()];
        dirs.extend(
            self.parent_workspaces
                .iter()
                .map(|workspace| workspace.segment_dir().to_path_buf()),
        );
        if let Some(workspace) = &self.mutation_workspace {
            dirs.push(workspace.segment_dir().to_path_buf());
        }
        for dir in dirs {
            typed.extend(property_index::scan_data_dir(&dir).unwrap_or_default());
            global.extend(property_index::scan_global_data_dir(&dir).unwrap_or_default());
        }
        typed.retain(|key| {
            !self.removed_property_indexes.contains(key)
                && !self.legacy_invalidated_property_indexes.contains(key)
        });
        global.retain(|property| !self.legacy_invalidated_global_indexes.contains(property));
        (typed.into_iter().collect(), global.into_iter().collect())
    }

    /// Rebuild every persistent bundle the graph has moved under, so the next
    /// lookup can serve from it again.
    ///
    /// `force` rebuilds even a bundle that is already current — what
    /// `reindex()` means. Without it, only the stale ones are rewritten, which
    /// is what keeps an unmutated `save()` at its previous cost.
    ///
    /// Callers: [`DirGraph::reindex`] and the pre-save consolidation. Both run
    /// under `&mut`, which is the reason the read path can only decline: a
    /// rebuild writes four files per bundle.
    pub(crate) fn refresh_persistent_indexes(&mut self, force: bool) -> std::io::Result<usize> {
        let node_bound = self.node_slot_len() as u32;
        let (typed, global) = self.persisted_index_names();
        let mut rebuilt = 0;
        for (node_type, property) in typed {
            let key = (node_type.clone(), property.clone());
            if !force && self.index_freshness.typed_is_fresh(&key, node_bound) {
                continue;
            }
            self.build_property_index(&node_type, &property)?;
            rebuilt += 1;
        }
        for property in global {
            if !force && self.index_freshness.global_is_fresh(&property, node_bound) {
                continue;
            }
            self.build_global_property_index(&property)?;
            rebuilt += 1;
        }
        // The baseline is deliberately left where it was. It stands in for
        // bundles this pass could *not* reach — the legacy-named ones, whose
        // filenames destroyed their identity — and those are still as stale as
        // they were, so a lookup that discovers one later must still decline.
        Ok(rebuilt)
    }

    /// Mask persisted lookup bundles that were built from raw legacy values.
    /// The immutable selected generation is untouched; a later explicit index
    /// build clears the matching mask after rebuilding against normalized data.
    pub(crate) fn invalidate_legacy_value_indexes(
        &mut self,
        typed: impl IntoIterator<Item = (String, String)>,
        global: impl IntoIterator<Item = String>,
    ) {
        for key in typed {
            self.property_indexes
                .write()
                .unwrap()
                .insert(key.clone(), None);
            self.legacy_invalidated_property_indexes.insert(key);
        }
        for property in global {
            self.global_indexes
                .write()
                .unwrap()
                .insert(property.clone(), None);
            self.legacy_invalidated_global_indexes.insert(property);
        }
    }
}
