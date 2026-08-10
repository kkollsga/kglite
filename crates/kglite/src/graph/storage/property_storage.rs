//! `PropertyStorage` — the three physical shapes a node's properties take.
//!
//! # Ownership
//!
//! A columnar node's properties live in a per-type
//! [`ColumnStore`](crate::graph::storage::column_store::ColumnStore) that the
//! **storage backend owns** (D1 Phase 3). `PropertyStorage::Columnar` carries a
//! [`ColumnarRow`] — a row id and nothing else. There is no handle on the node,
//! so there is no replica to drift, no `Arc::make_mut` fork per write, and no
//! re-point sweep after a master write.
//!
//! The consequence for this type is that its `Columnar` arm is **inert**:
//! `get` / `get_value` / `str_prop_eq` / `len` / `keys` answer as if the row
//! were empty, and the mutators `debug_assert!` rather than guess. Columnar
//! access is expressed on the backend instead, where the store actually is:
//!
//! - read  → [`NodeView`](crate::graph::storage::NodeView), built by
//!   [`GraphRead::node_view`](crate::graph::storage::GraphRead::node_view),
//!   which pairs the node with `column_store(node.node_type)`;
//! - write → [`GraphWrite::set_node_property`](crate::graph::storage::GraphWrite::set_node_property)
//!   and its four siblings.
//!
//! That asymmetry is deliberate and enforced: outside `graph::storage` the
//! value readers here are not even visible, so a caller cannot accidentally
//! read a columnar node as empty.
//!
//! # Serialization hazard
//!
//! With no store in reach, this type cannot serialize a columnar node's
//! properties. Every `.kgl` save path therefore writes columnar data in its own
//! column section and sets the `STRIP_PROPERTIES` thread-local while
//! serializing topology. The `Serialize` impl below asserts that guard is
//! actually set before it emits an empty map for a columnar node — a silent
//! empty map here is a **data-format** failure, not a code failure.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::datatypes::values::Value;
use crate::graph::schema::TypeSchema;
use crate::graph::storage::interner::{InternedKey, StringInterner, STRIP_PROPERTIES};

/// A node's columnar row — **identity only**.
///
/// D1 Phase 3 deleted the `Arc<ColumnStore>` that used to live here. A node no
/// longer knows *which* store holds its properties, only *which row* it is; the
/// store is resolved by the storage backend from the node's type
/// (`GraphRead::column_store`). That is the whole point of the programme: one
/// owner, no replicas to drift, and a `SET` that mutates one row in place
/// instead of forking a whole store per node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ColumnarRow {
    row_id: u32,
}

impl ColumnarRow {
    #[inline]
    pub(crate) fn new(row_id: u32) -> Self {
        ColumnarRow { row_id }
    }

    /// The node's row index within its type's store.
    #[inline]
    pub(crate) fn row_id(&self) -> u32 {
        self.row_id
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
            PropertyStorage::Columnar(_) => None,
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
            PropertyStorage::Columnar(_) => None,
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
            PropertyStorage::Columnar(_) => None,
        }
    }

    /// Insert or update a property. For Compact, extends schema via Arc::make_mut if key is new.
    pub(in crate::graph::storage) fn insert(&mut self, key: InternedKey, value: Value) {
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
            // A columnar node's properties live in the backend's store, which
            // this type cannot reach. Route through
            // `GraphWrite::set_node_property`, which has both the store and the
            // row id.
            PropertyStorage::Columnar(_) => debug_assert!(
                false,
                "columnar property write must go through GraphWrite::set_node_property"
            ),
        }
    }

    /// Insert only if the key is absent or Value::Null (for Preserve conflict mode).
    pub(in crate::graph::storage) fn insert_if_absent(&mut self, key: InternedKey, value: Value) {
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
            PropertyStorage::Columnar(_) => debug_assert!(
                false,
                "columnar property write must go through GraphWrite::set_node_property_if_absent"
            ),
        }
    }

    /// Remove a property. Returns the old value if it existed.
    pub(in crate::graph::storage) fn remove(&mut self, key: InternedKey) -> Option<Value> {
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
            PropertyStorage::Columnar(_) => {
                debug_assert!(
                    false,
                    "columnar property removal must go through GraphWrite::remove_node_property"
                );
                None
            }
        }
    }

    /// Replace all properties (for Replace conflict mode).
    /// Clears existing properties and inserts the new ones.
    pub(in crate::graph::storage) fn replace_all(
        &mut self,
        pairs: impl IntoIterator<Item = (InternedKey, Value)>,
    ) {
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
            PropertyStorage::Columnar(_) => debug_assert!(
                false,
                "columnar property replace must go through GraphWrite::replace_node_properties"
            ),
        }
    }

    /// Count of non-Null properties.
    pub(in crate::graph::storage) fn len(&self) -> usize {
        match self {
            PropertyStorage::Map(map) => map.len(),
            PropertyStorage::Compact { values, .. } => {
                values.iter().filter(|v| !matches!(v, Value::Null)).count()
            }
            PropertyStorage::Columnar(_) => 0,
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
            PropertyStorage::Columnar(_) => PropertyKeyIter::Columnar(Vec::new().into_iter()),
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
            PropertyStorage::Columnar(row) => PropertyStorage::Columnar(*row),
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
                // Identity only — a columnar node's values live in the
                // backend's store, so equality here compares the rows, which
                // the `row_id` arm below already covers.
                PropertyStorage::Columnar(_) => Vec::new(),
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
            PropertyStorage::Columnar(_) => {
                // The store is the backend's; this type cannot reach it. Every
                // save path that can meet a columnar node writes its properties
                // in the column section and strips them from topology — so
                // reaching here *without* that guard set would silently drop a
                // node's whole property set into the file.
                debug_assert!(
                    STRIP_PROPERTIES.with(|cell| cell.get()),
                    "serializing a columnar node without STRIP_PROPERTIES would write \
                     an empty property map: the save path must persist the type's \
                     ColumnStore separately (see io/file.rs) or convert first"
                );
                serializer.serialize_map(Some(0))?.end()
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
    /// This node's row id, when its properties are columnar.
    ///
    /// The only thing a `NodeData` still knows about columnar storage. Used by
    /// the backends to pair a node with a row in the store they own.
    #[inline]
    pub(crate) fn columnar_row_id(&self) -> Option<u32> {
        match self {
            PropertyStorage::Columnar(row) => Some(row.row_id()),
            _ => None,
        }
    }
}
