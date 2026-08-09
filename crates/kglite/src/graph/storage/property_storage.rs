//! `PropertyStorage` — the three physical shapes a node's properties take, and
//! the boundary that keeps a columnar node's store handle inside the storage
//! layer.
//!
//! # Why this lives under `graph::storage`
//!
//! A columnar node's properties live in a per-type
//! [`ColumnStore`](crate::graph::storage::column_store::ColumnStore) that the
//! storage backend owns. Before D1, `PropertyStorage::Columnar` exposed its
//! `Arc<ColumnStore>` as a public-in-crate enum-variant field, so *any* module
//! could pattern-match a node and read one replica of that store — which is how
//! the two shipped defects in `dev-docs/plans/d1-column-store-ownership.md` §2
//! arose, and what made the caller inventory necessary in the first place.
//!
//! The handle now sits behind [`ColumnarRow`], whose `store` field is private to
//! `graph::storage`. Outside the storage layer there is exactly **one** way to
//! reach it — [`ColumnarRow::node_handle`] — and the set of call sites is pinned
//! by `column_ownership_tests::the_node_handle_escape_has_exactly_the_phase_3_call_sites`.
//! Everything else reads through [`NodeView`](crate::graph::storage::NodeView),
//! which resolves the store the backend answers with.
//!
//! Reading a property value through `node_handle()` is always wrong. It is for
//! handle *identity* (`Arc::ptr_eq` drift checks), handle *re-pointing* (the
//! refresh sweep, rollback restore) and lifecycle bookkeeping — the operations
//! D1 Phase 3 deletes outright.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::datatypes::values::Value;
use crate::graph::schema::TypeSchema;
use crate::graph::storage::column_store::ColumnStore;
use crate::graph::storage::interner::{InternedKey, StringInterner, STRIP_PROPERTIES};

/// A node's columnar row: *which* store, and *which* row inside it.
///
/// `store` is private to `graph::storage`; see the module docs for the single
/// allowlisted escape.
pub(crate) struct ColumnarRow {
    /// The node-held replica of the type's column store.
    ///
    /// D1 Phase 3 deletes this field: the backend becomes the sole owner and a
    /// node keeps only `row_id`. Until then it is reachable outside
    /// `graph::storage` only through [`ColumnarRow::node_handle`].
    store: Arc<ColumnStore>,
    row_id: u32,
}

impl ColumnarRow {
    #[inline]
    pub(crate) fn new(store: Arc<ColumnStore>, row_id: u32) -> Self {
        ColumnarRow { store, row_id }
    }

    /// The node's row index within its type's store.
    #[inline]
    pub(crate) fn row_id(&self) -> u32 {
        self.row_id
    }

    /// **The one allowlisted escape.** Returns the node-held store handle.
    ///
    /// Every caller outside `graph::storage` is, by construction, on the D1
    /// Phase-3 work list — this is the only route to a node's own `Arc`, and
    /// `column_ownership_tests::the_node_handle_escape_has_exactly_the_phase_3_call_sites`
    /// pins the exact set. Adding a call site fails that test.
    ///
    /// **Never use this to read a property value.** Use
    /// [`NodeView`](crate::graph::storage::NodeView), which resolves the store
    /// the backend owns rather than this replica.
    #[inline]
    pub(crate) fn node_handle(&self) -> &Arc<ColumnStore> {
        &self.store
    }

    /// **Allowlisted escape (mutating).** Re-point this node at `store`.
    ///
    /// The refresh sweep (`columnar_write::refresh_columnar_node_handles`) and
    /// the rollback restore arm are the only callers, and both disappear in D1
    /// Phase 3 along with the field itself. Pinned by the same inventory test
    /// as [`ColumnarRow::node_handle`].
    #[inline]
    pub(crate) fn repoint(&mut self, store: Arc<ColumnStore>) {
        self.store = store;
    }

    /// Storage-internal mutable handle, for the copy-on-write mutators below.
    #[inline]
    fn store_mut(&mut self) -> &mut Arc<ColumnStore> {
        &mut self.store
    }

    /// Storage-internal read handle.
    #[inline]
    pub(in crate::graph::storage) fn store(&self) -> &ColumnStore {
        &self.store
    }
}

impl Clone for ColumnarRow {
    fn clone(&self) -> Self {
        ColumnarRow {
            store: Arc::clone(&self.store),
            row_id: self.row_id,
        }
    }
}

/// Compact property storage for nodes.
/// - `Map`: transient during deserialization (before compaction).
/// - `Compact`: steady state with a shared `TypeSchema` and dense `Vec<Value>`.
/// - `Columnar`: column-oriented storage via a shared `ColumnStore`.
pub(crate) enum PropertyStorage {
    /// HashMap storage (used during deserialization, before `compact_properties()`).
    Map(HashMap<InternedKey, Value>),
    /// Slot-vec storage indexed by shared TypeSchema.
    /// `Value::Null` in a slot means "property absent".
    Compact {
        schema: Arc<TypeSchema>,
        values: Vec<Value>,
    },
    /// Column-oriented storage — properties live in a per-type `ColumnStore`
    /// the backend owns. See [`ColumnarRow`] for why the handle is not a bare
    /// field here.
    Columnar(ColumnarRow),
}

/// Zero-allocation iterator over property key strings.
/// Replaces the prior `Box<dyn Iterator>` returned by `PropertyStorage::keys`,
/// saving one heap allocation per call (keys/iter runs in the hot path of
/// `keys(n)` and `RETURN n {.*}` per row).
pub(crate) enum PropertyKeyIter<'a> {
    Map {
        inner: std::collections::hash_map::Keys<'a, InternedKey, Value>,
        interner: &'a StringInterner,
    },
    Compact {
        slots: &'a [InternedKey],
        values: &'a [Value],
        slot_idx: usize,
        interner: &'a StringInterner,
    },
    Columnar(std::vec::IntoIter<&'a str>),
}

impl<'a> Iterator for PropertyKeyIter<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        match self {
            PropertyKeyIter::Map { inner, interner } => inner.next().map(|k| interner.resolve(*k)),
            PropertyKeyIter::Compact {
                slots,
                values,
                slot_idx,
                interner,
            } => loop {
                let i = *slot_idx;
                if i >= slots.len() {
                    return None;
                }
                *slot_idx = i + 1;
                if values.get(i).is_some_and(|v| !matches!(v, Value::Null)) {
                    return Some(interner.resolve(slots[i]));
                }
            },
            PropertyKeyIter::Columnar(iter) => iter.next(),
        }
    }
}

impl PropertyStorage {
    /// Look up a property value by interned key. Returns None if absent or Value::Null.
    ///
    /// Returns `Cow::Borrowed` for Map/Compact variants (zero-copy).
    /// Future Columnar variant will return `Cow::Owned`.
    #[inline]
    pub(in crate::graph::storage) fn get(&self, key: InternedKey) -> Option<Cow<'_, Value>> {
        match self {
            PropertyStorage::Map(map) => map.get(&key).map(Cow::Borrowed),
            PropertyStorage::Compact { schema, values } => schema
                .slot(key)
                .and_then(|slot| values.get(slot as usize))
                .filter(|v| !matches!(v, Value::Null))
                .map(Cow::Borrowed),
            PropertyStorage::Columnar(row) => row.store().get(row.row_id(), key).map(Cow::Owned),
        }
    }

    /// Look up a property value by interned key, returning an owned Value.
    /// More efficient than `get()` for callers that always need ownership
    /// (avoids Cow wrapping/unwrapping overhead).
    #[inline]
    pub(in crate::graph::storage) fn get_value(&self, key: InternedKey) -> Option<Value> {
        match self {
            PropertyStorage::Map(map) => map.get(&key).cloned(),
            PropertyStorage::Compact { schema, values } => schema
                .slot(key)
                .and_then(|slot| values.get(slot as usize))
                .filter(|v| !matches!(v, Value::Null))
                .cloned(),
            PropertyStorage::Columnar(row) => row.store().get(row.row_id(), key),
        }
    }

    /// Check if a property exists (non-Null).
    #[inline]
    pub(in crate::graph::storage) fn contains(&self, key: InternedKey) -> bool {
        self.get(key).is_some()
    }

    /// Zero-allocation string equality for a property against `target`.
    ///
    /// For columnar storage this bypasses the `Value::String(s.to_string())`
    /// materialisation in `get()`, which dominates string-equality scans on
    /// mapped graphs. For the non-columnar variants the cost is already
    /// borrowable, so we just wrap the existing `get`.
    #[inline]
    pub(in crate::graph::storage) fn str_prop_eq(
        &self,
        key: InternedKey,
        target: &str,
    ) -> Option<bool> {
        match self {
            PropertyStorage::Map(map) => map
                .get(&key)
                .map(|v| matches!(v, Value::String(s) if s == target)),
            PropertyStorage::Compact { schema, values } => schema
                .slot(key)
                .and_then(|slot| values.get(slot as usize))
                .filter(|v| !matches!(v, Value::Null))
                .map(|v| matches!(v, Value::String(s) if s == target)),
            PropertyStorage::Columnar(row) => row.store().str_prop_eq(row.row_id(), key, target),
        }
    }

    /// Presence check on storage the caller **already owns mutably**, for a
    /// read-modify-write that must not go through the backend.
    ///
    /// The one caller is `add_nodes`' stub-promotion step, which holds
    /// `&mut NodeData` and is deciding whether to clear a marker it is about to
    /// write. Returns presence only — never a value. **Reading a property value
    /// off a `NodeData` is what `NodeView` is for**; the value readers on this
    /// type are deliberately confined to `graph::storage`.
    #[inline]
    pub(crate) fn contains_own_key(&self, key: InternedKey) -> bool {
        self.contains(key)
    }

    /// Insert or update a property. For Compact, extends schema via Arc::make_mut if key is new.
    pub fn insert(&mut self, key: InternedKey, value: Value) {
        match self {
            PropertyStorage::Map(map) => {
                map.insert(key, value);
            }
            PropertyStorage::Compact { schema, values } => {
                let slot = if let Some(s) = schema.slot(key) {
                    s as usize
                } else {
                    // New key: extend schema
                    let s = Arc::make_mut(schema).add_key(key) as usize;
                    s
                };
                if slot >= values.len() {
                    values.resize(slot + 1, Value::Null);
                }
                values[slot] = value;
            }
            PropertyStorage::Columnar(row) => {
                let rid = row.row_id();
                Arc::make_mut(row.store_mut()).set(rid, key, &value, None);
            }
        }
    }

    /// Insert only if the key is absent or Value::Null (for Preserve conflict mode).
    pub fn insert_if_absent(&mut self, key: InternedKey, value: Value) {
        match self {
            PropertyStorage::Map(map) => {
                map.entry(key).or_insert(value);
            }
            PropertyStorage::Compact { schema, values } => {
                if let Some(slot) = schema.slot(key) {
                    let slot = slot as usize;
                    if slot < values.len() {
                        if matches!(values[slot], Value::Null) {
                            values[slot] = value;
                        }
                        // else: existing non-Null value, preserve it
                    } else {
                        // Slot beyond current Vec: insert
                        values.resize(slot + 1, Value::Null);
                        values[slot] = value;
                    }
                } else {
                    // Key not in schema: extend and insert
                    let slot = Arc::make_mut(schema).add_key(key) as usize;
                    if slot >= values.len() {
                        values.resize(slot + 1, Value::Null);
                    }
                    values[slot] = value;
                }
            }
            PropertyStorage::Columnar(row) => {
                let rid = row.row_id();
                if row.store().get(rid, key).is_none() {
                    Arc::make_mut(row.store_mut()).set(rid, key, &value, None);
                }
            }
        }
    }

    /// Remove a property. Returns the old value if it existed.
    pub fn remove(&mut self, key: InternedKey) -> Option<Value> {
        match self {
            PropertyStorage::Map(map) => map.remove(&key),
            PropertyStorage::Compact { schema, values } => schema.slot(key).and_then(|slot| {
                let slot = slot as usize;
                if slot < values.len() {
                    let old = std::mem::replace(&mut values[slot], Value::Null);
                    if matches!(old, Value::Null) {
                        None
                    } else {
                        Some(old)
                    }
                } else {
                    None
                }
            }),
            PropertyStorage::Columnar(row) => {
                let rid = row.row_id();
                let old = row.store().get(rid, key);
                if old.is_some() {
                    Arc::make_mut(row.store_mut()).set(rid, key, &Value::Null, None);
                }
                old
            }
        }
    }

    /// Replace all properties (for Replace conflict mode).
    /// Clears existing properties and inserts the new ones.
    pub fn replace_all(&mut self, pairs: impl IntoIterator<Item = (InternedKey, Value)>) {
        match self {
            PropertyStorage::Map(map) => {
                map.clear();
                map.extend(pairs);
            }
            PropertyStorage::Compact { schema, values } => {
                // Reset all slots to Null
                for v in values.iter_mut() {
                    *v = Value::Null;
                }
                for (key, value) in pairs {
                    let slot = if let Some(s) = schema.slot(key) {
                        s as usize
                    } else {
                        Arc::make_mut(schema).add_key(key) as usize
                    };
                    if slot >= values.len() {
                        values.resize(slot + 1, Value::Null);
                    }
                    values[slot] = value;
                }
            }
            PropertyStorage::Columnar(row) => {
                let rid = row.row_id();
                let st = Arc::make_mut(row.store_mut());
                // Clear existing properties by setting all to null
                let props: Vec<_> = st.row_properties(rid).into_iter().map(|(k, _)| k).collect();
                for k in props {
                    st.set(rid, k, &Value::Null, None);
                }
                // Insert new pairs
                for (key, value) in pairs {
                    st.set(rid, key, &value, None);
                }
            }
        }
    }

    /// Count of non-Null properties.
    pub(in crate::graph::storage) fn len(&self) -> usize {
        match self {
            PropertyStorage::Map(map) => map.len(),
            PropertyStorage::Compact { values, .. } => {
                values.iter().filter(|v| !matches!(v, Value::Null)).count()
            }
            PropertyStorage::Columnar(row) => row.store().row_properties(row.row_id()).len(),
        }
    }

    /// Iterate over property keys as strings. Requires interner for resolution.
    /// Drain all property (InternedKey, Value) pairs out of this storage.
    /// Used by mapped mode to push properties into a ColumnStore.
    /// After this call, self is left as an empty Map.
    pub fn drain_to_interned_pairs(
        &mut self,
        _interner: &StringInterner,
    ) -> Vec<(InternedKey, Value)> {
        match std::mem::replace(self, PropertyStorage::Map(HashMap::new())) {
            PropertyStorage::Map(map) => map.into_iter().collect(),
            PropertyStorage::Compact { schema, values } => schema
                .slots
                .iter()
                .zip(values)
                .filter(|(_, v)| !matches!(v, Value::Null))
                .map(|(ik, v)| (*ik, v))
                .collect(),
            PropertyStorage::Columnar { .. } => {
                // Already columnar — nothing to drain
                Vec::new()
            }
        }
    }

    pub(in crate::graph::storage) fn keys<'a>(
        &'a self,
        interner: &'a StringInterner,
    ) -> PropertyKeyIter<'a> {
        match self {
            PropertyStorage::Map(map) => PropertyKeyIter::Map {
                inner: map.keys(),
                interner,
            },
            PropertyStorage::Compact { schema, values } => PropertyKeyIter::Compact {
                slots: &schema.slots,
                values,
                slot_idx: 0,
                interner,
            },
            PropertyStorage::Columnar(row) => {
                // Columnar can't borrow keys through the enum — materialize once.
                let props = row.store().row_properties(row.row_id());
                let keys: Vec<&'a str> = props
                    .iter()
                    .filter_map(|(ik, _)| interner.try_resolve(*ik))
                    .collect();
                PropertyKeyIter::Columnar(keys.into_iter())
            }
        }
    }

    /// Build Compact storage from pre-interned key-value pairs and a shared schema.
    pub fn from_compact(
        pairs: impl IntoIterator<Item = (InternedKey, Value)>,
        schema: &Arc<TypeSchema>,
    ) -> Self {
        let mut values = vec![Value::Null; schema.len()];
        for (key, value) in pairs {
            if let Some(slot) = schema.slot(key) {
                values[slot as usize] = value;
            }
        }
        PropertyStorage::Compact {
            schema: Arc::clone(schema),
            values,
        }
    }
}

impl Clone for PropertyStorage {
    fn clone(&self) -> Self {
        match self {
            PropertyStorage::Map(map) => PropertyStorage::Map(map.clone()),
            PropertyStorage::Compact { schema, values } => PropertyStorage::Compact {
                schema: Arc::clone(schema),
                values: values.clone(),
            },
            PropertyStorage::Columnar(row) => PropertyStorage::Columnar(row.clone()),
        }
    }
}

impl std::fmt::Debug for PropertyStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PropertyStorage::Map(map) => f.debug_tuple("Map").field(map).finish(),
            PropertyStorage::Compact { values, .. } => {
                f.debug_tuple("Compact").field(values).finish()
            }
            PropertyStorage::Columnar(row) => f
                .debug_struct("Columnar")
                .field("row_id", &row.row_id())
                .finish(),
        }
    }
}

impl PartialEq for PropertyStorage {
    fn eq(&self, other: &Self) -> bool {
        // Compare logical content: same set of (InternedKey, non-Null Value) pairs.
        // This is only used in tests (NodeData derives PartialEq).
        fn collect_entries(ps: &PropertyStorage) -> Vec<(InternedKey, Value)> {
            match ps {
                PropertyStorage::Map(map) => {
                    let mut entries: Vec<_> = map.iter().map(|(&k, v)| (k, v.clone())).collect();
                    entries.sort_by_key(|(k, _)| k.as_u64());
                    entries
                }
                PropertyStorage::Compact { schema, values } => {
                    let mut entries: Vec<_> = schema
                        .slots
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &ik)| {
                            values.get(i).and_then(|v| {
                                if matches!(v, Value::Null) {
                                    None
                                } else {
                                    Some((ik, v.clone()))
                                }
                            })
                        })
                        .collect();
                    entries.sort_by_key(|(k, _)| k.as_u64());
                    entries
                }
                PropertyStorage::Columnar(row) => {
                    let mut entries: Vec<_> = row.store().row_properties(row.row_id());
                    entries.sort_by_key(|(k, _)| k.as_u64());
                    entries
                }
            }
        }
        collect_entries(self) == collect_entries(other)
    }
}

impl Serialize for PropertyStorage {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        // v3 topology mode: serialize empty map to strip node properties
        if STRIP_PROPERTIES.with(|cell| cell.get()) {
            return serializer.serialize_map(Some(0))?.end();
        }
        match self {
            PropertyStorage::Map(map) => map.serialize(serializer),
            PropertyStorage::Compact { schema, values } => {
                // Count non-Null entries for accurate map length
                let count = values.iter().filter(|v| !matches!(v, Value::Null)).count();
                let mut map_ser = serializer.serialize_map(Some(count))?;
                for (i, ik) in schema.slots.iter().enumerate() {
                    if let Some(v) = values.get(i) {
                        if !matches!(v, Value::Null) {
                            map_ser.serialize_entry(ik, v)?;
                        }
                    }
                }
                map_ser.end()
            }
            PropertyStorage::Columnar(row) => {
                // Materialize properties from column store for serialization
                let props = row.store().row_properties(row.row_id());
                let mut map_ser = serializer.serialize_map(Some(props.len()))?;
                for (ik, v) in &props {
                    map_ser.serialize_entry(ik, v)?;
                }
                map_ser.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for PropertyStorage {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let map = HashMap::<InternedKey, Value>::deserialize(deserializer)?;
        Ok(PropertyStorage::Map(map))
    }
}

impl PropertyStorage {
    /// Columnar id column for this row, if the properties are columnar.
    ///
    /// `NodeData::id` delegates here for the mapped-mode `Value::Null`
    /// sentinel, so `graph::schema` never reaches a store handle.
    #[inline]
    pub(crate) fn columnar_id(&self) -> Option<Value> {
        match self {
            PropertyStorage::Columnar(row) => row.store().get_id(row.row_id()),
            _ => None,
        }
    }

    /// Columnar title column for this row, if the properties are columnar.
    #[inline]
    pub(crate) fn columnar_title(&self) -> Option<Value> {
        match self {
            PropertyStorage::Columnar(row) => row.store().get_title(row.row_id()),
            _ => None,
        }
    }
}
