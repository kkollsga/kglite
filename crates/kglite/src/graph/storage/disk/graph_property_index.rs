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
    pub fn build_property_index(&self, node_type: &str, property: &str) -> std::io::Result<usize> {
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

        let node_bound = self.node_slots.len();
        let mut entries: Vec<(String, u32)> = Vec::with_capacity(node_bound);
        for i in 0..node_bound {
            let nslot = self.node_slots.get(i);
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
        let idx =
            property_index::PropertyIndex::build(&self.data_dir, node_type, property, entries)?;
        self.property_indexes.write().unwrap().insert(
            (node_type.to_string(), property.to_string()),
            Some(Arc::new(idx)),
        );
        Ok(count)
    }

    /// Exact-match lookup. Returns `None` when no index has been built
    /// for `(node_type, property)`; returns `Some(Vec)` (possibly empty)
    /// when an index exists. The planner uses the distinction to decide
    /// whether to route through the fast path or fall back to scan.
    pub fn lookup_property_eq(
        &self,
        node_type: &str,
        property: &str,
        value: &str,
    ) -> Option<Vec<NodeIndex>> {
        let key = (node_type.to_string(), property.to_string());
        // Fast path: cached handle.
        {
            let read = self.property_indexes.read().unwrap();
            if let Some(slot) = read.get(&key) {
                return slot.as_ref().map(|idx| idx.lookup_eq_str(value));
            }
        }
        // Slow path: check disk. If files exist, mmap and cache.
        let idx_opt = property_index::PropertyIndex::open(&self.data_dir, node_type, property)
            .ok()
            .flatten();
        let result = idx_opt.as_ref().map(|idx| idx.lookup_eq_str(value));
        self.property_indexes
            .write()
            .unwrap()
            .insert(key, idx_opt.map(Arc::new));
        result
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
        let key = (node_type.to_string(), property.to_string());
        {
            let read = self.property_indexes.read().unwrap();
            if let Some(slot) = read.get(&key) {
                return slot
                    .as_ref()
                    .map(|idx| idx.lookup_prefix_str(prefix, limit));
            }
        }
        let idx_opt = property_index::PropertyIndex::open(&self.data_dir, node_type, property)
            .ok()
            .flatten();
        let result = idx_opt
            .as_ref()
            .map(|idx| idx.lookup_prefix_str(prefix, limit));
        self.property_indexes
            .write()
            .unwrap()
            .insert(key, idx_opt.map(Arc::new));
        result
    }

    /// Whether an index has been built for `(node_type, property)`.
    /// Checks the cache first, then the filesystem.
    pub fn has_property_index(&self, node_type: &str, property: &str) -> bool {
        let key = (node_type.to_string(), property.to_string());
        if let Some(slot) = self.property_indexes.read().unwrap().get(&key) {
            return slot.is_some();
        }
        let (meta, _, _, _) = property_index::file_paths(&self.data_dir, node_type, property);
        meta.exists()
    }

    /// Build a cross-type global index for `property`. Scans every
    /// alive `DiskNodeSlot` and emits one `(string_value, NodeIndex)`
    /// entry per node where `property` resolves to a non-empty string
    /// (regular column, title alias, or id alias — same resolution
    /// order as [`build_property_index`]).
    ///
    /// Powers untyped patterns like `MATCH (n {label: 'X'})` and the
    /// `search(text)` helper. Re-run whenever the graph is rebuilt.
    pub fn build_global_property_index(&self, property: &str) -> std::io::Result<usize> {
        let prop_key = InternedKey::from_str(property);
        let node_bound = self.node_slots.len();
        let mut entries: Vec<(String, u32)> = Vec::with_capacity(node_bound / 2);

        // Cache per-type (column_store, schema_slot) lookups so every
        // node in the same type reuses the slot resolution.
        type ColStore = Arc<crate::graph::storage::column_store::ColumnStore>;
        type TypeCacheEntry = Option<(ColStore, Option<u16>)>;
        let mut type_cache: HashMap<u64, TypeCacheEntry> = HashMap::new();

        for i in 0..node_bound {
            let nslot = self.node_slots.get(i);
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
        let idx = property_index::PropertyIndex::build_global(&self.data_dir, property, entries)?;
        self.global_indexes
            .write()
            .unwrap()
            .insert(property.to_string(), Some(Arc::new(idx)));
        Ok(count)
    }

    /// Exact-match lookup across every node type for a cross-type
    /// global index. Returns `None` when no index has been built for
    /// `property`; returns `Some(Vec)` (possibly empty) otherwise.
    pub fn lookup_global_eq(&self, property: &str, value: &str) -> Option<Vec<NodeIndex>> {
        {
            let read = self.global_indexes.read().unwrap();
            if let Some(slot) = read.get(property) {
                return slot.as_ref().map(|idx| idx.lookup_eq_str(value));
            }
        }
        let idx_opt = property_index::PropertyIndex::open_global(&self.data_dir, property)
            .ok()
            .flatten();
        let result = idx_opt.as_ref().map(|idx| idx.lookup_eq_str(value));
        self.global_indexes
            .write()
            .unwrap()
            .insert(property.to_string(), idx_opt.map(Arc::new));
        result
    }

    /// Prefix lookup (STARTS WITH) against the cross-type global
    /// index. Same `None`/`Some` semantics as [`lookup_global_eq`].
    pub fn lookup_global_prefix(
        &self,
        property: &str,
        prefix: &str,
        limit: usize,
    ) -> Option<Vec<NodeIndex>> {
        {
            let read = self.global_indexes.read().unwrap();
            if let Some(slot) = read.get(property) {
                return slot
                    .as_ref()
                    .map(|idx| idx.lookup_prefix_str(prefix, limit));
            }
        }
        let idx_opt = property_index::PropertyIndex::open_global(&self.data_dir, property)
            .ok()
            .flatten();
        let result = idx_opt
            .as_ref()
            .map(|idx| idx.lookup_prefix_str(prefix, limit));
        self.global_indexes
            .write()
            .unwrap()
            .insert(property.to_string(), idx_opt.map(Arc::new));
        result
    }
}
