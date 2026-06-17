// src/graph/schema.rs
use crate::datatypes::values::{FilterCondition, Value};
use crate::graph::storage::interner::STRIP_PROPERTIES;
pub use crate::graph::storage::interner::{InternedKey, StringInterner};
pub(crate) use crate::graph::storage::interner::{
    SerdeDeserializeGuard, SerdeSerializeGuard, StripPropertiesGuard,
};
use crate::graph::storage::GraphRead;

// Phase 9 split: DirGraph + GraphBackend moved to sibling modules.
// Re-exported here to preserve `crate::graph::schema::X` import paths.
pub use crate::graph::dir_graph::DirGraph;
pub use crate::graph::storage::backend::GraphBackend;
// MemoryGraph re-export: required by `storage/recording.rs` tests.
// DO NOT REMOVE even if cargo fix suggests it — the test-only usage is
// what keeps this file's API stable.
#[allow(unused_imports)]
pub use crate::graph::storage::{MappedGraph, MemoryGraph};
use petgraph::graph::NodeIndex;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Reserved property name marking an auto-vivified stub node — one
/// created to satisfy an edge whose endpoint had no row of its own.
/// Set to `true` at vivification; cleared (`Null`) when the real node
/// row later upserts the node ("promotion"). `purge_provisional()`
/// deletes whatever is still marked. Reserved: a user-supplied
/// property of this name is rejected at blueprint validation.
pub const PROVISIONAL_KEY: &str = "_provisional";

// ─── Type Schema & Compact Property Storage ──────────────────────────────────

/// Shared schema for all nodes of one type — maps property keys to dense slot indices.
/// All nodes of the same type share an `Arc<TypeSchema>`, keeping per-node overhead to 8 bytes.
#[derive(Debug, Clone)]
pub struct TypeSchema {
    /// slot_index → interned key (for iteration / serialization)
    pub(crate) slots: Vec<InternedKey>,
    /// interned key → slot_index (for O(1) lookup). FxHash, not the std
    /// SipHasher: `InternedKey` is already a well-distributed FNV `u64`, so a
    /// cryptographic hash is pure overhead. `slot()` is the per-property,
    /// per-row, per-column lookup on the Compact (production) read path —
    /// samply (2026-05-29) showed SipHash here at ~23% of in-memory query CPU.
    key_to_slot: FxHashMap<InternedKey, u16>,
}

impl TypeSchema {
    /// Create an empty schema.
    pub fn new() -> Self {
        TypeSchema {
            slots: Vec::new(),
            key_to_slot: FxHashMap::default(),
        }
    }

    /// Build a schema from an iterator of interned keys.
    pub fn from_keys(keys: impl IntoIterator<Item = InternedKey>) -> Self {
        let mut schema = TypeSchema::new();
        for key in keys {
            if !schema.key_to_slot.contains_key(&key) {
                let slot = schema.slots.len() as u16;
                schema.slots.push(key);
                schema.key_to_slot.insert(key, slot);
            }
        }
        schema
    }

    /// Get the slot index for a key, or None if not in schema.
    #[inline]
    pub fn slot(&self, key: InternedKey) -> Option<u16> {
        self.key_to_slot.get(&key).copied()
    }

    /// Number of slots in the schema.
    #[inline]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Create a new schema containing all keys from both schemas.
    pub fn merge(&self, other: &TypeSchema) -> TypeSchema {
        let mut merged = self.clone();
        for &key in &other.slots {
            merged.add_key(key);
        }
        merged
    }

    /// Add a new key to the schema. Returns the new slot index.
    /// If the key already exists, returns the existing slot index.
    pub fn add_key(&mut self, key: InternedKey) -> u16 {
        if let Some(&slot) = self.key_to_slot.get(&key) {
            slot
        } else {
            let slot = self.slots.len() as u16;
            self.slots.push(key);
            self.key_to_slot.insert(key, slot);
            slot
        }
    }

    /// Iterate over all (slot_index, interned_key) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (u16, InternedKey)> + '_ {
        self.slots.iter().enumerate().map(|(i, &k)| (i as u16, k))
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
    /// Column-oriented storage — properties live in a per-type `ColumnStore`.
    /// The node's row is identified by `row_id`.
    Columnar {
        store: Arc<crate::graph::storage::column_store::ColumnStore>,
        row_id: u32,
    },
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

/// Zero-allocation iterator over (key_string, &Value) property pairs.
/// Same rationale as `PropertyKeyIter` — saves the per-call heap alloc for
/// the prior `Box<dyn Iterator>`.
pub(crate) enum PropertyIter<'a> {
    Map {
        inner: std::collections::hash_map::Iter<'a, InternedKey, Value>,
        interner: &'a StringInterner,
    },
    Compact {
        slots: &'a [InternedKey],
        values: &'a [Value],
        slot_idx: usize,
        interner: &'a StringInterner,
    },
    /// Columnar storage can't produce `&'a Value` references since values aren't
    /// stored as owned `Value`s — callers use `iter_owned()` instead. This variant
    /// exists only so the enum shape matches across all `PropertyStorage` cases.
    Columnar,
}

impl<'a> Iterator for PropertyIter<'a> {
    type Item = (&'a str, &'a Value);

    #[inline]
    fn next(&mut self) -> Option<(&'a str, &'a Value)> {
        match self {
            PropertyIter::Map { inner, interner } => {
                inner.next().map(|(k, v)| (interner.resolve(*k), v))
            }
            PropertyIter::Compact {
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
                if let Some(v) = values.get(i) {
                    if !matches!(v, Value::Null) {
                        return Some((interner.resolve(slots[i]), v));
                    }
                }
            },
            PropertyIter::Columnar => None,
        }
    }
}

impl PropertyStorage {
    /// Look up a property value by interned key. Returns None if absent or Value::Null.
    ///
    /// Returns `Cow::Borrowed` for Map/Compact variants (zero-copy).
    /// Future Columnar variant will return `Cow::Owned`.
    #[inline]
    pub fn get(&self, key: InternedKey) -> Option<Cow<'_, Value>> {
        match self {
            PropertyStorage::Map(map) => map.get(&key).map(Cow::Borrowed),
            PropertyStorage::Compact { schema, values } => schema
                .slot(key)
                .and_then(|slot| values.get(slot as usize))
                .filter(|v| !matches!(v, Value::Null))
                .map(Cow::Borrowed),
            PropertyStorage::Columnar { store, row_id } => store.get(*row_id, key).map(Cow::Owned),
        }
    }

    /// Look up a property value by interned key, returning an owned Value.
    /// More efficient than `get()` for callers that always need ownership
    /// (avoids Cow wrapping/unwrapping overhead).
    #[inline]
    pub fn get_value(&self, key: InternedKey) -> Option<Value> {
        match self {
            PropertyStorage::Map(map) => map.get(&key).cloned(),
            PropertyStorage::Compact { schema, values } => schema
                .slot(key)
                .and_then(|slot| values.get(slot as usize))
                .filter(|v| !matches!(v, Value::Null))
                .cloned(),
            PropertyStorage::Columnar { store, row_id } => store.get(*row_id, key),
        }
    }

    /// Check if a property exists (non-Null).
    #[inline]
    pub fn contains(&self, key: InternedKey) -> bool {
        self.get(key).is_some()
    }

    /// Zero-allocation string equality for a property against `target`.
    ///
    /// For columnar storage this bypasses the `Value::String(s.to_string())`
    /// materialisation in `get()`, which dominates string-equality scans on
    /// mapped graphs. For the non-columnar variants the cost is already
    /// borrowable, so we just wrap the existing `get`.
    #[inline]
    pub fn str_prop_eq(&self, key: InternedKey, target: &str) -> Option<bool> {
        match self {
            PropertyStorage::Map(map) => map
                .get(&key)
                .map(|v| matches!(v, Value::String(s) if s == target)),
            PropertyStorage::Compact { schema, values } => schema
                .slot(key)
                .and_then(|slot| values.get(slot as usize))
                .filter(|v| !matches!(v, Value::Null))
                .map(|v| matches!(v, Value::String(s) if s == target)),
            PropertyStorage::Columnar { store, row_id } => store.str_prop_eq(*row_id, key, target),
        }
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
            PropertyStorage::Columnar { store, row_id } => {
                Arc::make_mut(store).set(*row_id, key, &value, None);
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
            PropertyStorage::Columnar { store, row_id } => {
                if store.get(*row_id, key).is_none() {
                    Arc::make_mut(store).set(*row_id, key, &value, None);
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
            PropertyStorage::Columnar { store, row_id } => {
                let old = store.get(*row_id, key);
                if old.is_some() {
                    Arc::make_mut(store).set(*row_id, key, &Value::Null, None);
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
            PropertyStorage::Columnar { store, row_id } => {
                let st = Arc::make_mut(store);
                // Clear existing properties by setting all to null
                let props: Vec<_> = st
                    .row_properties(*row_id)
                    .into_iter()
                    .map(|(k, _)| k)
                    .collect();
                for k in props {
                    st.set(*row_id, k, &Value::Null, None);
                }
                // Insert new pairs
                for (key, value) in pairs {
                    st.set(*row_id, key, &value, None);
                }
            }
        }
    }

    /// Count of non-Null properties.
    pub fn len(&self) -> usize {
        match self {
            PropertyStorage::Map(map) => map.len(),
            PropertyStorage::Compact { values, .. } => {
                values.iter().filter(|v| !matches!(v, Value::Null)).count()
            }
            PropertyStorage::Columnar { store, row_id } => store.row_properties(*row_id).len(),
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

    pub fn keys<'a>(&'a self, interner: &'a StringInterner) -> PropertyKeyIter<'a> {
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
            PropertyStorage::Columnar { store, row_id } => {
                // Columnar can't borrow keys through the enum — materialize once.
                let props = store.row_properties(*row_id);
                let keys: Vec<&'a str> = props
                    .iter()
                    .filter_map(|(ik, _)| interner.try_resolve(*ik))
                    .collect();
                PropertyKeyIter::Columnar(keys.into_iter())
            }
        }
    }

    /// Iterate over (key_string, &Value) pairs. Requires interner for resolution.
    /// For Map/Compact, yields borrowed references. For Columnar, yields nothing —
    /// callers that need columnar property pairs should use `iter_owned()`.
    pub fn iter<'a>(&'a self, interner: &'a StringInterner) -> PropertyIter<'a> {
        match self {
            PropertyStorage::Map(map) => PropertyIter::Map {
                inner: map.iter(),
                interner,
            },
            PropertyStorage::Compact { schema, values } => PropertyIter::Compact {
                slots: &schema.slots,
                values,
                slot_idx: 0,
                interner,
            },
            PropertyStorage::Columnar { .. } => PropertyIter::Columnar,
        }
    }

    /// Iterate over (key_string, Value) pairs for Columnar storage.
    /// Returns owned values. Works for all variants.
    pub fn iter_owned<'a>(&'a self, interner: &'a StringInterner) -> Vec<(String, Value)> {
        match self {
            PropertyStorage::Map(map) => map
                .iter()
                .map(|(k, v)| (interner.resolve(*k).to_string(), v.clone()))
                .collect(),
            PropertyStorage::Compact { schema, values } => schema
                .slots
                .iter()
                .enumerate()
                .filter_map(|(i, ik)| {
                    values.get(i).and_then(|v| {
                        if matches!(v, Value::Null) {
                            None
                        } else {
                            Some((interner.resolve(*ik).to_string(), v.clone()))
                        }
                    })
                })
                .collect(),
            PropertyStorage::Columnar { store, row_id } => store
                .row_properties(*row_id)
                .into_iter()
                .filter_map(|(ik, v)| interner.try_resolve(ik).map(|s| (s.to_string(), v)))
                .collect(),
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
            PropertyStorage::Columnar { store, row_id } => PropertyStorage::Columnar {
                store: Arc::clone(store),
                row_id: *row_id,
            },
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
            PropertyStorage::Columnar { row_id, .. } => {
                f.debug_struct("Columnar").field("row_id", row_id).finish()
            }
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
                PropertyStorage::Columnar { store, row_id } => {
                    let mut entries: Vec<_> = store.row_properties(*row_id);
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
            PropertyStorage::Columnar { store, row_id } => {
                // Materialize properties from column store for serialization
                let props = store.row_properties(*row_id);
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

/// Spatial configuration for a node type. Declares which properties hold
/// spatial data (lat/lon pairs, WKT geometries) and enables auto-resolution
/// in Cypher `distance(a, b)` and fluent API methods.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct SpatialConfig {
    /// Primary lat/lon location: (lat_field, lon_field). At most one per type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub location: Option<(String, String)>,
    /// Primary WKT geometry field name. At most one per type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry: Option<String>,
    /// Named lat/lon points: name → (lat_field, lon_field). Zero or more.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub points: HashMap<String, (String, String)>,
    /// Named WKT shape fields: name → field_name. Zero or more.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub shapes: HashMap<String, String>,
}

/// Parse spatial column-type entries from the wheel's `column_types`
/// dict shape. Recognizes `location.lat`, `location.lon`, `geometry`,
/// `point.<name>.lat`, `point.<name>.lon`, `shape.<name>`. Returns
/// `(Some(config), cleaned_pairs)` if any spatial entries were found,
/// `(None, original_pairs)` otherwise. The cleaned pairs replace
/// recognized spatial type strings with their natural storage types
/// (`float` for lat/lon, `str` for WKT shapes) so downstream
/// dataframe loaders treat them correctly.
///
/// Lifted from kglite-py's `parse_spatial_column_types` in 0.10.1 —
/// the wheel keeps only the `Bound<PyDict>` → `Vec<(String, String)>`
/// extraction wrapper.
/// Result of column-type parsing: optional config + cleaned (col_name, type_str) pairs.
pub type SpatialColumnParseResult = (Option<SpatialConfig>, Vec<(String, String)>);

pub fn parse_spatial_column_types_from_pairs(
    pairs: Vec<(String, String)>,
) -> Result<SpatialColumnParseResult, String> {
    let mut cleaned: Vec<(String, String)> = Vec::with_capacity(pairs.len());
    let mut config = SpatialConfig::default();
    let mut has_spatial = false;

    // Track partial location/point entries (need both lat and lon).
    let mut location_lat: Option<String> = None;
    let mut location_lon: Option<String> = None;
    let mut point_lats: HashMap<String, String> = HashMap::new();
    let mut point_lons: HashMap<String, String> = HashMap::new();

    for (col_name, type_str) in pairs {
        let type_lower = type_str.to_lowercase();
        match type_lower.as_str() {
            "location.lat" => {
                location_lat = Some(col_name.clone());
                cleaned.push((col_name, "float".to_string()));
                has_spatial = true;
            }
            "location.lon" => {
                location_lon = Some(col_name.clone());
                cleaned.push((col_name, "float".to_string()));
                has_spatial = true;
            }
            "geometry" => {
                config.geometry = Some(col_name.clone());
                cleaned.push((col_name, "str".to_string()));
                has_spatial = true;
            }
            _ if type_lower.starts_with("point.") => {
                let parts: Vec<&str> = type_lower.splitn(3, '.').collect();
                if parts.len() == 3 {
                    let name = parts[1].to_string();
                    match parts[2] {
                        "lat" => {
                            point_lats.insert(name, col_name.clone());
                        }
                        "lon" => {
                            point_lons.insert(name, col_name.clone());
                        }
                        _ => {
                            return Err(format!(
                                "Invalid spatial type '{}' for column '{}'. \
                                 Expected 'point.<name>.lat' or 'point.<name>.lon'.",
                                type_str, col_name
                            ));
                        }
                    }
                    cleaned.push((col_name, "float".to_string()));
                    has_spatial = true;
                } else {
                    return Err(format!(
                        "Invalid spatial type '{}' for column '{}'. \
                         Expected 'point.<name>.lat' or 'point.<name>.lon'.",
                        type_str, col_name
                    ));
                }
            }
            _ if type_lower.starts_with("shape.") => {
                let parts: Vec<&str> = type_lower.splitn(2, '.').collect();
                if parts.len() == 2 {
                    let name = parts[1].to_string();
                    config.shapes.insert(name, col_name.clone());
                    cleaned.push((col_name, "str".to_string()));
                    has_spatial = true;
                } else {
                    return Err(format!(
                        "Invalid spatial type '{}' for column '{}'.",
                        type_str, col_name
                    ));
                }
            }
            _ => {
                cleaned.push((col_name, type_str));
            }
        }
    }

    if !has_spatial {
        return Ok((None, cleaned));
    }

    match (location_lat, location_lon) {
        (Some(lat), Some(lon)) => config.location = Some((lat, lon)),
        (Some(_), None) | (None, Some(_)) => {
            return Err(
                "Incomplete location: both 'location.lat' and 'location.lon' must be specified."
                    .to_string(),
            );
        }
        (None, None) => {}
    }

    let all_point_names: std::collections::HashSet<&String> =
        point_lats.keys().chain(point_lons.keys()).collect();
    for name in all_point_names {
        match (point_lats.get(name), point_lons.get(name)) {
            (Some(lat), Some(lon)) => {
                config
                    .points
                    .insert(name.clone(), (lat.clone(), lon.clone()));
            }
            _ => {
                return Err(format!(
                    "Incomplete point '{}': both 'point.{}.lat' and 'point.{}.lon' must be specified.",
                    name, name, name
                ));
            }
        }
    }

    Ok((Some(config), cleaned))
}

/// Temporal configuration for a node type or connection type.
/// Declares which properties hold validity-period dates (valid_from, valid_to).
/// When configured, temporal filtering is applied automatically in
/// `select()` (for nodes) and `traverse()` (for connections).
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct TemporalConfig {
    /// Property name holding the start date, e.g. "fldLicenseeFrom" or "date_from"
    pub valid_from: String,
    /// Property name holding the end date, e.g. "fldLicenseeTo" or "date_to"
    pub valid_to: String,
}

/// Parse temporal column-type entries from the wheel's `column_types`
/// dict shape. Recognizes `validFrom` / `validTo` (case-insensitive).
/// Returns `(Some(config), cleaned_pairs)` if BOTH validFrom and
/// validTo are found; `(None, original_pairs)` if neither is found.
/// Returns `Err` if exactly one is found (asymmetric config is a
/// data-shape mistake — better to fail loudly).
///
/// Lifted from kglite-py's `parse_temporal_column_types` in 0.10.1.
/// Result of temporal column-type parsing: optional config + cleaned pairs.
pub type TemporalColumnParseResult = (Option<TemporalConfig>, Vec<(String, String)>);

pub fn parse_temporal_column_types_from_pairs(
    pairs: Vec<(String, String)>,
) -> Result<TemporalColumnParseResult, String> {
    let mut cleaned: Vec<(String, String)> = Vec::with_capacity(pairs.len());
    let mut valid_from_col: Option<String> = None;
    let mut valid_to_col: Option<String> = None;

    for (col_name, type_str) in pairs {
        let type_lower = type_str.to_lowercase();
        match type_lower.as_str() {
            "validfrom" => {
                valid_from_col = Some(col_name.clone());
                cleaned.push((col_name, "datetime".to_string()));
            }
            "validto" => {
                valid_to_col = Some(col_name.clone());
                cleaned.push((col_name, "datetime".to_string()));
            }
            _ => {
                cleaned.push((col_name, type_str));
            }
        }
    }

    match (valid_from_col, valid_to_col) {
        (Some(from), Some(to)) => Ok((
            Some(TemporalConfig {
                valid_from: from,
                valid_to: to,
            }),
            cleaned,
        )),
        (Some(_), None) | (None, Some(_)) => Err(
            "Incomplete temporal config: both 'validFrom' and 'validTo' column types must be specified."
                .to_string(),
        ),
        (None, None) => Ok((None, cleaned)),
    }
}

/// Per-type ID index. Uses compact u32 keys when all IDs are UniqueId,
/// falling back to general Value keys otherwise.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum TypeIdIndex {
    /// All IDs are UniqueId(u32) — compact, ~8 bytes per entry.
    Integer(HashMap<u32, NodeIndex>),
    /// Mixed ID types — general, ~60 bytes per entry.
    General(HashMap<Value, NodeIndex>),
}

impl TypeIdIndex {
    /// Look up a node by ID value, with type coercion.
    pub fn get(&self, id: &Value) -> Option<NodeIndex> {
        match self {
            TypeIdIndex::Integer(map) => match id {
                Value::UniqueId(u) => map.get(u).copied(),
                Value::Int64(i) => {
                    if *i >= 0 && *i <= u32::MAX as i64 {
                        map.get(&(*i as u32)).copied()
                    } else {
                        None
                    }
                }
                Value::Float64(f) => {
                    if f.fract() == 0.0 {
                        let i = *f as i64;
                        if i >= 0 && i <= u32::MAX as i64 {
                            map.get(&(i as u32)).copied()
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                // NB: no string→u32 coercion. A `String` id queried against a
                // UniqueId-keyed index does NOT match (e.g. `{id:'a1'}` must
                // not resolve to `UniqueId(1)`). Datasets that expose a
                // string id form (e.g. Wikidata `Q76`) store it as a queryable
                // property — query `{nid:'Q76'}`, not `{id:'Q76'}`.
                _ => None,
            },
            TypeIdIndex::General(map) => {
                if let Some(&idx) = map.get(id) {
                    return Some(idx);
                }
                // Type coercion fallback
                match id {
                    Value::Int64(i) => {
                        if *i >= 0 && *i <= u32::MAX as i64 {
                            map.get(&Value::UniqueId(*i as u32)).copied()
                        } else {
                            None
                        }
                    }
                    Value::UniqueId(u) => map.get(&Value::Int64(*u as i64)).copied(),
                    Value::Float64(f) => {
                        if f.fract() == 0.0 {
                            let i = *f as i64;
                            if let Some(&idx) = map.get(&Value::Int64(i)) {
                                return Some(idx);
                            }
                            if i >= 0 && i <= u32::MAX as i64 {
                                return map.get(&Value::UniqueId(i as u32)).copied();
                            }
                        }
                        None
                    }
                    // String ids match only by exact value (handled above);
                    // no prefix-strip coercion. See the Integer arm.
                    _ => None,
                }
            }
        }
    }

    /// Insert an ID → NodeIndex mapping.
    pub fn insert(&mut self, id: Value, idx: NodeIndex) {
        match self {
            TypeIdIndex::Integer(map) => {
                if let Value::UniqueId(u) = id {
                    map.insert(u, idx);
                } else {
                    // Demote to General
                    let mut general: HashMap<Value, NodeIndex> =
                        map.drain().map(|(k, v)| (Value::UniqueId(k), v)).collect();
                    general.insert(id, idx);
                    *self = TypeIdIndex::General(general);
                }
            }
            TypeIdIndex::General(map) => {
                map.insert(id, idx);
            }
        }
    }

    /// Iterate over all (Value, NodeIndex) pairs.
    pub fn iter(&self) -> Box<dyn Iterator<Item = (Value, NodeIndex)> + '_> {
        match self {
            TypeIdIndex::Integer(map) => {
                Box::new(map.iter().map(|(&k, &v)| (Value::UniqueId(k), v)))
            }
            TypeIdIndex::General(map) => Box::new(map.iter().map(|(k, &v)| (k.clone(), v))),
        }
    }
}

impl Default for TypeIdIndex {
    fn default() -> Self {
        TypeIdIndex::General(HashMap::new())
    }
}

/// Lightweight snapshot of a node's data: id, title, type, and properties.
/// Used as the return type for node queries and traversals.
#[derive(Clone, Debug)]
pub struct NodeInfo {
    pub id: Value,
    pub title: Value,
    pub node_type: String,
    pub properties: HashMap<String, Value>,
}

/// Records a filtering, sorting, or traversal operation applied to a selection.
/// Used by `explain()` to show the query execution plan.
#[derive(Clone, Debug)]
pub enum SelectionOperation {
    Filter(HashMap<String, FilterCondition>),
    Sort(Vec<(String, bool)>), // (field_name, ascending)
    Traverse {
        connection_type: String,
        direction: Option<String>,
        max_nodes: Option<usize>,
    },
    Custom(String), // For operations that don't fit other categories
}

/// A single level in the selection hierarchy — holds node sets grouped
/// by their parent (for traversals) and tracks applied operations.
#[derive(Clone, Debug)]
pub struct SelectionLevel {
    pub selections: HashMap<Option<NodeIndex>, Vec<NodeIndex>>, // parent_idx -> selected_children
    pub operations: Vec<SelectionOperation>,
}

impl SelectionLevel {
    pub fn new() -> Self {
        SelectionLevel {
            selections: HashMap::new(),
            operations: Vec::new(),
        }
    }

    pub fn add_selection(&mut self, parent: Option<NodeIndex>, children: Vec<NodeIndex>) {
        self.selections.insert(parent, children);
    }

    pub fn get_all_nodes(&self) -> Vec<NodeIndex> {
        self.selections
            .values()
            .flat_map(|children| children.iter().copied())
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.selections.is_empty()
    }

    pub fn iter_groups(&self) -> impl Iterator<Item = (&Option<NodeIndex>, &Vec<NodeIndex>)> {
        self.selections.iter()
    }

    /// Returns an iterator over all node indices without allocating a Vec.
    /// Use this instead of get_all_nodes() when you only need to iterate or count.
    pub fn iter_node_indices(&self) -> impl Iterator<Item = NodeIndex> + '_ {
        self.selections
            .values()
            .flat_map(|children| children.iter().copied())
    }

    /// Returns the total count of nodes without allocating a Vec.
    /// More efficient than get_all_nodes().len() for just getting the count.
    pub fn node_count(&self) -> usize {
        self.selections.values().map(|v| v.len()).sum()
    }
}

/// Represents a single step in the query execution plan
#[derive(Clone, Debug)]
pub struct PlanStep {
    pub operation: String,
    pub node_type: Option<String>,
    pub estimated_rows: usize,
    pub actual_rows: Option<usize>,
}

impl PlanStep {
    pub fn new(operation: &str, node_type: Option<&str>, estimated_rows: usize) -> Self {
        PlanStep {
            operation: operation.to_string(),
            node_type: node_type.map(|s| s.to_string()),
            estimated_rows,
            actual_rows: None,
        }
    }

    pub fn with_actual_rows(mut self, actual: usize) -> Self {
        self.actual_rows = Some(actual);
        self
    }
}

/// Tracks the current selection state across a chain of query operations
/// (type_filter → filter → traverse → ...). Supports nested levels for
/// parent-child traversals and records execution plan steps for `explain()`.
#[derive(Clone, Default)]
pub struct CurrentSelection {
    levels: Vec<SelectionLevel>,
    current_level: usize,
    execution_plan: Vec<PlanStep>,
}

impl CurrentSelection {
    pub fn new() -> Self {
        let mut selection = CurrentSelection {
            levels: Vec::new(),
            current_level: 0,
            execution_plan: Vec::new(),
        };
        selection.add_level(); // Always start with an initial level
        selection
    }

    pub fn add_level(&mut self) {
        // No need to pass level index
        self.levels.push(SelectionLevel::new());
        self.current_level = self.levels.len() - 1;
    }

    pub fn clear(&mut self) {
        self.levels.clear();
        self.current_level = 0;
        self.execution_plan.clear();
        self.add_level(); // Ensure we always have at least one level after clearing
    }

    /// Add a step to the execution plan
    pub fn add_plan_step(&mut self, step: PlanStep) {
        self.execution_plan.push(step);
    }

    /// Get the execution plan steps
    pub fn get_execution_plan(&self) -> &[PlanStep] {
        &self.execution_plan
    }

    /// Clear just the execution plan (for fresh queries)
    pub fn clear_execution_plan(&mut self) {
        self.execution_plan.clear();
    }

    pub fn get_level_count(&self) -> usize {
        self.levels.len()
    }

    pub fn get_level(&self, index: usize) -> Option<&SelectionLevel> {
        self.levels.get(index)
    }

    pub fn get_level_mut(&mut self, index: usize) -> Option<&mut SelectionLevel> {
        self.levels.get_mut(index)
    }

    /// Returns the node count for the current (most recent) level without allocation.
    pub fn current_node_count(&self) -> usize {
        self.levels.last().map(|l| l.node_count()).unwrap_or(0)
    }

    /// Returns true if any filtering/selection operation has been applied to the current level.
    /// Used to distinguish "no filter applied" (pristine state) from "filter returned 0 results".
    pub fn has_active_selection(&self) -> bool {
        self.levels
            .last()
            .map(|l| !l.operations.is_empty())
            .unwrap_or(false)
    }

    /// Returns an iterator over node indices in the current (most recent) level.
    pub fn current_node_indices(&self) -> impl Iterator<Item = NodeIndex> + '_ {
        self.levels
            .last()
            .into_iter()
            .flat_map(|l| l.iter_node_indices())
    }

    /// Returns the node type of the first node in the current selection, if any.
    /// Used by spatial auto-resolution to look up SpatialConfig.
    pub fn first_node_type(&self, graph: &DirGraph) -> Option<String> {
        self.current_node_indices()
            .next()
            .and_then(|idx| graph.graph.node_weight(idx))
            .map(|node| node.node_type_str(&graph.interner).to_string())
    }
}

/// Copy-on-write wrapper for CurrentSelection.
/// Avoids cloning the selection on every method call when the selection isn't modified.
/// Implements Deref/DerefMut for transparent usage where CurrentSelection is expected.
#[derive(Clone, Default)]
pub struct CowSelection {
    inner: Arc<CurrentSelection>,
}

impl CowSelection {
    pub fn new() -> Self {
        CowSelection {
            inner: Arc::new(CurrentSelection::new()),
        }
    }
}

// Implement Deref for transparent read access
impl std::ops::Deref for CowSelection {
    type Target = CurrentSelection;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

// Implement DerefMut for copy-on-write mutation
impl std::ops::DerefMut for CowSelection {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.inner)
    }
}

/// Key for single-property indexes: (node_type, property_name)
pub type IndexKey = (String, String);

/// Key for composite indexes: (node_type, property_names)
pub type CompositeIndexKey = (String, Vec<String>);

/// Composite value key: tuple of values for multi-field lookup
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CompositeValue(pub Vec<Value>);

/// Metadata stamped into saved files for version tracking and portability warnings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SaveMetadata {
    /// Format version — incremented when DirGraph layout changes.
    /// 0 = files saved before this field existed (via serde default).
    /// 1 = first versioned format.
    pub format_version: u32,
    /// Library version at save time, e.g. "0.4.7".
    pub library_version: String,
}

impl SaveMetadata {
    pub fn current() -> Self {
        SaveMetadata {
            format_version: 2,
            library_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// Type connectivity triple: one row of the type-level graph.
/// (source_type) -[connection_type]-> (target_type) with edge count.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectivityTriple {
    pub src: String,
    pub conn: String,
    pub tgt: String,
    pub count: usize,
}

/// Metadata about a connection type: which node types it connects and what properties it carries.
#[derive(Debug, Clone, Default)]
pub struct ConnectionTypeInfo {
    pub source_types: HashSet<String>,
    pub target_types: HashSet<String>,
    /// property_name → type_string (e.g. "weight" → "Float64")
    pub property_types: HashMap<String, String>,
}

/// Custom serializer emits sorted keys for the two HashSet<String> and the
/// HashMap<String, String> so that `.kgl` v3 saves stay byte-deterministic
/// regardless of per-run HashMap seed. Phase 4's golden-hash test pinned the
/// current digest; Phase 5 hardened the invariant so richer fixtures
/// (multiple source/target types or property keys) don't slip through.
impl Serialize for ConnectionTypeInfo {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut sorted_sources: Vec<&String> = self.source_types.iter().collect();
        sorted_sources.sort();
        let mut sorted_targets: Vec<&String> = self.target_types.iter().collect();
        sorted_targets.sort();
        let mut sorted_props: Vec<(&String, &String)> = self.property_types.iter().collect();
        sorted_props.sort_by(|a, b| a.0.cmp(b.0));
        let property_types: std::collections::BTreeMap<&String, &String> =
            sorted_props.into_iter().collect();

        let mut state = serializer.serialize_struct("ConnectionTypeInfo", 3)?;
        state.serialize_field("source_types", &sorted_sources)?;
        state.serialize_field("target_types", &sorted_targets)?;
        state.serialize_field("property_types", &property_types)?;
        state.end()
    }
}

/// Custom deserializer to handle both old format (source_type/target_type as single strings)
/// and new format (source_types/target_types as sets).
impl<'de> Deserialize<'de> for ConnectionTypeInfo {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Legacy {
            source_type: Option<String>,
            target_type: Option<String>,
            #[serde(default)]
            source_types: Option<HashSet<String>>,
            #[serde(default)]
            target_types: Option<HashSet<String>>,
            #[serde(default)]
            property_types: HashMap<String, String>,
        }

        let legacy = Legacy::deserialize(deserializer)?;
        let source_types = legacy.source_types.unwrap_or_else(|| {
            legacy
                .source_type
                .map(|s| HashSet::from([s]))
                .unwrap_or_default()
        });
        let target_types = legacy.target_types.unwrap_or_else(|| {
            legacy
                .target_type
                .map(|s| HashSet::from([s]))
                .unwrap_or_default()
        });
        Ok(ConnectionTypeInfo {
            source_types,
            target_types,
            property_types: legacy.property_types,
        })
    }
}

/// Contiguous columnar storage for f32 embeddings associated with a (node_type, property_name).
/// All vectors in one store share the same dimensionality.
/// The flat Vec<f32> layout enables SIMD-friendly linear scans during vector search.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmbeddingStore {
    /// Embedding dimensionality (e.g. 384, 768, 1536)
    pub dimension: usize,
    /// Contiguous f32 buffer: embedding i occupies data[i*dimension..(i+1)*dimension]
    pub data: Vec<f32>,
    /// Maps NodeIndex.index() -> slot position in the contiguous buffer
    pub node_to_slot: HashMap<usize, usize>,
    /// Reverse map: slot -> NodeIndex.index(), needed for returning results
    pub slot_to_node: Vec<usize>,
    /// Default distance metric for this embedding store (e.g. "cosine", "poincare").
    /// Used when no explicit metric is provided at query time.
    #[serde(default)]
    pub metric: Option<String>,
    /// Identity of the model that produced these vectors (e.g. "BAAI/bge-m3").
    /// Set from the embedder's `model_id()` during `embed_texts`; `None` for
    /// vectors supplied directly via `add_embeddings`/`set_embeddings` or by a
    /// backend that doesn't name its model. Surfaced via `embedding_info()`.
    #[serde(default)]
    pub model_id: Option<String>,
    /// Per-node content hash of the embedded text (`NodeIndex.index()` → hash),
    /// stamped during `embed_texts`. Lets `embed_texts(mode='changed')`
    /// re-embed exactly the nodes whose text changed since the last pass,
    /// instead of every node or only the missing ones. Empty for stores built
    /// from raw vectors (no source text to hash).
    #[serde(default)]
    pub text_hashes: HashMap<usize, u64>,
    /// Cached L2 norm (‖v‖) per slot, parallel to `slot_to_node` — `norms[slot]`
    /// is the Euclidean norm of the vector at `data[slot*dimension..]`. A derived
    /// acceleration structure: cosine search reads it instead of recomputing the
    /// stored vector's norm on every query (collapsing cosine's inner loop to a
    /// single dot product). Deliberately NOT serialized (`#[serde(skip)]`) — it
    /// is fully determined by `data`, so persisting it would bloat the `.kgl`
    /// file and risk drift. Rebuilt from `data` after load via `rebuild_norms`.
    #[serde(skip)]
    pub norms: Vec<f32>,
    /// Optional HNSW approximate-nearest-neighbour index over this store's
    /// vectors, built on demand via `build_vector_index`. When present (and the
    /// query isn't `exact`), top-k vector search dispatches through it instead of
    /// a full linear scan. Indexes nodes by *slot*, so any change to the slot
    /// layout (in-place vector replacement, `add_embeddings`, compaction)
    /// invalidates it — callers drop it via `invalidate_index`. NOT serialized
    /// here (`#[serde(skip)]`): it rides in a dedicated versioned, skippable
    /// sub-section of the `.kgl` so the on-disk format can evolve independently;
    /// absent that section it's simply rebuilt.
    #[serde(skip)]
    pub index: Option<crate::graph::algorithms::hnsw::HnswIndex>,
}

/// Squared L2 norm of a vector, computed with the same 4-accumulator
/// `chunks_exact(8)` pattern the cosine kernel uses for the stored vector, so a
/// cached norm matches the value cosine would have computed inline (within fp
/// rounding). Auto-vectorizes (SSE2/AVX2/NEON) on the hot rebuild path.
#[inline]
fn l2_norm_sq(v: &[f32]) -> f32 {
    let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let chunks = v.chunks_exact(8);
    let rem = chunks.remainder();
    for c in chunks {
        s0 += c[0] * c[0];
        s1 += c[1] * c[1];
        s2 += c[2] * c[2];
        s3 += c[3] * c[3];
        s0 += c[4] * c[4];
        s1 += c[5] * c[5];
        s2 += c[6] * c[6];
        s3 += c[7] * c[7];
    }
    for &x in rem {
        s0 += x * x;
    }
    (s0 + s1) + (s2 + s3)
}

impl EmbeddingStore {
    pub fn new(dimension: usize) -> Self {
        EmbeddingStore {
            dimension,
            data: Vec::new(),
            node_to_slot: HashMap::new(),
            slot_to_node: Vec::new(),
            metric: None,
            model_id: None,
            text_hashes: HashMap::new(),
            norms: Vec::new(),
            index: None,
        }
    }

    pub fn with_metric(dimension: usize, metric: &str) -> Self {
        EmbeddingStore {
            dimension,
            data: Vec::new(),
            node_to_slot: HashMap::new(),
            slot_to_node: Vec::new(),
            metric: Some(metric.to_string()),
            model_id: None,
            text_hashes: HashMap::new(),
            norms: Vec::new(),
            index: None,
        }
    }

    /// FNV-1a 64-bit hash of an embedded text — the content fingerprint
    /// stored in `text_hashes` for change detection. Deterministic across
    /// processes (same rationale as `InternedKey::from_str`), so a hash
    /// persisted in one process matches a re-hash in another after reload.
    pub fn text_hash(text: &str) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;
        let mut h = FNV_OFFSET;
        for &byte in text.as_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
        h
    }

    /// Record the source-text hash for a node (by `NodeIndex.index()`).
    pub fn set_text_hash(&mut self, node_index: usize, hash: u64) {
        self.text_hashes.insert(node_index, hash);
    }

    /// `true` if the node has no embedding yet, or its stored text hash
    /// differs from `current_hash` (the text changed since it was embedded),
    /// or it was embedded without a recorded hash. Drives `mode='changed'`.
    pub fn is_stale(&self, node_index: usize, current_hash: u64) -> bool {
        if !self.node_to_slot.contains_key(&node_index) {
            return true;
        }
        match self.text_hashes.get(&node_index) {
            Some(&stored) => stored != current_hash,
            None => true,
        }
    }

    /// Add or replace an embedding for a node. Returns the slot index.
    /// Keeps the cached L2 norm (`norms[slot]`) in sync with `data`, and drops
    /// any HNSW index — mutating the vectors/slot layout would invalidate its
    /// topology (rebuild via `build_index` after a batch of edits).
    pub fn set_embedding(&mut self, node_index: usize, embedding: &[f32]) -> usize {
        self.index = None;
        let norm = l2_norm_sq(embedding).sqrt();
        if let Some(&slot) = self.node_to_slot.get(&node_index) {
            // Replace existing embedding in-place
            let start = slot * self.dimension;
            self.data[start..start + self.dimension].copy_from_slice(embedding);
            self.norms[slot] = norm;
            slot
        } else {
            // Append new embedding
            let slot = self.slot_to_node.len();
            self.node_to_slot.insert(node_index, slot);
            self.slot_to_node.push(node_index);
            self.data.extend_from_slice(embedding);
            self.norms.push(norm);
            slot
        }
    }

    /// Get the embedding slice for a node (by NodeIndex.index()).
    #[inline]
    pub fn get_embedding(&self, node_index: usize) -> Option<&[f32]> {
        self.node_to_slot.get(&node_index).map(|&slot| {
            let start = slot * self.dimension;
            &self.data[start..start + self.dimension]
        })
    }

    /// Get the embedding slice and its cached L2 norm for a node, in one lookup.
    /// The norm lets cosine scoring skip recomputing the stored vector's norm.
    #[inline]
    pub fn get_embedding_with_norm(&self, node_index: usize) -> Option<(&[f32], f32)> {
        self.node_to_slot.get(&node_index).map(|&slot| {
            let start = slot * self.dimension;
            (&self.data[start..start + self.dimension], self.norms[slot])
        })
    }

    /// Drop any HNSW index (it must be rebuilt after the vectors change).
    #[inline]
    pub fn invalidate_index(&mut self) {
        self.index = None;
    }

    /// `true` if an HNSW index is currently built over this store.
    #[inline]
    pub fn has_index(&self) -> bool {
        self.index.is_some()
    }

    /// Build (or rebuild) an HNSW index over this store's vectors for `metric`.
    /// Errors for Poincaré (unsupported by HNSW — stays brute-force). Ensures
    /// the norm cache is populated first (cosine navigation needs it).
    pub fn build_index(
        &mut self,
        metric: crate::graph::algorithms::vector::DistanceMetric,
        params: crate::graph::algorithms::hnsw::HnswParams,
        seed: u64,
    ) -> Result<(), String> {
        let hm = crate::graph::algorithms::hnsw::HnswMetric::from_distance(metric).ok_or_else(
            || "HNSW does not support the Poincaré metric; it stays on the exact (brute-force) path.".to_string(),
        )?;
        if self.norms.len() != self.slot_to_node.len() {
            self.rebuild_norms();
        }
        self.index = Some(crate::graph::algorithms::hnsw::HnswIndex::build(
            &self.data,
            &self.norms,
            self.dimension,
            hm,
            params,
            seed,
        ));
        Ok(())
    }

    /// Recompute every cached norm from `data`. Call after any wholesale
    /// replacement of `data` that bypasses `set_embedding` — deserialization
    /// (`norms` is `#[serde(skip)]`, so it loads empty) and slot remapping
    /// during compaction. Idempotent.
    pub fn rebuild_norms(&mut self) {
        let n = self.slot_to_node.len();
        self.norms.clear();
        self.norms.reserve(n);
        for slot in 0..n {
            let start = slot * self.dimension;
            self.norms
                .push(l2_norm_sq(&self.data[start..start + self.dimension]).sqrt());
        }
    }

    /// Number of stored embeddings.
    #[inline]
    pub fn len(&self) -> usize {
        self.slot_to_node.len()
    }
}
/// Structural conveniences KGLite exposes on a node (`n.label`, `n.type`,
/// `n.node_type`, `n.name`) that a user may also legitimately store as a real
/// property. For these the stored property WINS during resolution (KG-1,
/// 2026-05-30); the structural value is only the fallback used when no such
/// property exists. `id`/`title` are deliberately NOT soft aliases — they are
/// the node's identity fields and are always returned canonically (no stored
/// property can shadow them).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SoftAliasFallback {
    /// `name` falls back to the node title.
    Title,
    /// `type` / `node_type` / `label` fall back to the node type string.
    TypeString,
}

/// Classify a (post-alias) property name as a soft structural alias, or
/// `None` for a normal property / identity field. Single source of truth so
/// every resolution path (RETURN, WHERE, inline-map, EXISTS, disk fast path)
/// agrees on which names are property-first.
#[inline]
pub fn soft_alias_fallback(resolved: &str) -> Option<SoftAliasFallback> {
    match resolved {
        "name" => Some(SoftAliasFallback::Title),
        "type" | "node_type" | "label" => Some(SoftAliasFallback::TypeString),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeData {
    pub id: Value,
    pub title: Value,
    pub node_type: InternedKey,
    pub(crate) properties: PropertyStorage,
}

impl NodeData {
    /// Create a new NodeData, interning all property keys and the node type.
    /// Builds PropertyStorage::Map — call compact_properties() later to convert to Compact.
    pub fn new(
        id: Value,
        title: Value,
        node_type: String,
        properties: HashMap<String, Value>,
        interner: &mut StringInterner,
    ) -> Self {
        let type_key = interner.get_or_intern(&node_type);
        let interned_props = properties
            .into_iter()
            .map(|(k, v)| {
                let key = interner.get_or_intern(&k);
                (key, v)
            })
            .collect();
        NodeData {
            id,
            title,
            node_type: type_key,
            properties: PropertyStorage::Map(interned_props),
        }
    }

    /// Create a new NodeData with Compact storage using a pre-built schema.
    pub fn new_compact(
        id: Value,
        title: Value,
        node_type: String,
        properties: HashMap<String, Value>,
        interner: &mut StringInterner,
        schema: &Arc<TypeSchema>,
    ) -> Self {
        let type_key = interner.get_or_intern(&node_type);
        let pairs = properties.into_iter().map(|(k, v)| {
            let key = interner.get_or_intern(&k);
            (key, v)
        });
        NodeData {
            id,
            title,
            node_type: type_key,
            properties: PropertyStorage::from_compact(pairs, schema),
        }
    }

    /// Create a new NodeData with Compact storage from pre-interned keys (avoids re-interning).
    pub fn new_compact_preinterned(
        id: Value,
        title: Value,
        node_type: InternedKey,
        properties: Vec<(InternedKey, Value)>,
        schema: &Arc<TypeSchema>,
    ) -> Self {
        NodeData {
            id,
            title,
            node_type,
            properties: PropertyStorage::from_compact(properties, schema),
        }
    }

    /// Create a new NodeData with Map storage from pre-interned keys (avoids re-interning).
    pub fn new_preinterned(
        id: Value,
        title: Value,
        node_type: InternedKey,
        properties: Vec<(InternedKey, Value)>,
    ) -> Self {
        let map: HashMap<InternedKey, Value> = properties.into_iter().collect();
        NodeData {
            id,
            title,
            node_type,
            properties: PropertyStorage::Map(map),
        }
    }

    /// Get the node's ID. In mapped mode (Null sentinel), reads from ColumnStore.
    #[inline]
    pub fn id(&self) -> Cow<'_, Value> {
        if matches!(self.id, Value::Null) {
            if let PropertyStorage::Columnar { store, row_id } = &self.properties {
                if let Some(v) = store.get_id(*row_id) {
                    return Cow::Owned(v);
                }
            }
        }
        Cow::Borrowed(&self.id)
    }

    /// Get the node's title. In mapped mode (Null sentinel), reads from ColumnStore.
    #[inline]
    pub fn title(&self) -> Cow<'_, Value> {
        if matches!(self.title, Value::Null) {
            if let PropertyStorage::Columnar { store, row_id } = &self.properties {
                if let Some(v) = store.get_title(*row_id) {
                    return Cow::Owned(v);
                }
            }
        }
        Cow::Borrowed(&self.title)
    }

    /// Resolve the node type to a string. Requires the interner.
    #[inline]
    pub fn node_type_str<'a>(&self, interner: &'a StringInterner) -> &'a str {
        interner.resolve(self.node_type)
    }

    /// Returns a reference to the field value without cloning.
    /// Uses hash-based lookup — no interner needed.
    ///
    /// Returns `Cow::Borrowed` for in-memory storage (zero-copy).
    #[inline]
    pub fn get_field_ref(&self, field: &str) -> Option<Cow<'_, Value>> {
        match field {
            "id" => Some(self.id()),
            "title" => Some(self.title()),
            _ => self.properties.get(InternedKey::from_str(field)),
        }
    }

    /// Case-insensitive substring check on a string-typed field. Returns
    /// `false` if the field is missing or non-string. `needle_lower` is
    /// expected to already be lowercased — the function only lowercases
    /// the haystack. Lifted from kglite-py in 0.10.1 so all bindings
    /// share the same case-insensitive matcher.
    pub fn field_contains_ci(&self, field: &str, needle_lower: &str) -> bool {
        self.get_field_ref(field)
            .and_then(|v| match &*v {
                Value::String(s) => Some(s.to_lowercase().contains(needle_lower)),
                _ => None,
            })
            .unwrap_or(false)
    }

    /// Case-insensitive prefix check on a string-typed field. Returns
    /// `false` if the field is missing or non-string. `prefix_lower` is
    /// expected to already be lowercased.
    pub fn field_starts_with_ci(&self, field: &str, prefix_lower: &str) -> bool {
        self.get_field_ref(field)
            .and_then(|v| match &*v {
                Value::String(s) => Some(s.to_lowercase().starts_with(prefix_lower)),
                _ => None,
            })
            .unwrap_or(false)
    }

    /// Returns a property value (excludes id/title/type).
    /// Uses hash-based lookup — no interner needed.
    ///
    /// Returns `Cow::Borrowed` for in-memory storage (zero-copy).
    #[inline]
    pub fn get_property(&self, key: &str) -> Option<Cow<'_, Value>> {
        self.properties.get(InternedKey::from_str(key))
    }

    /// Like `get_property` but returns owned Value directly (no Cow overhead).
    /// Preferred in the Cypher executor hot path where ownership is always needed.
    #[inline]
    pub fn get_property_value(&self, key: &str) -> Option<Value> {
        self.properties.get_value(InternedKey::from_str(key))
    }

    /// Returns an iterator over property keys (excludes id/title/type).
    /// Requires interner to resolve InternedKey → &str.
    #[inline]
    pub fn property_keys<'a>(
        &'a self,
        interner: &'a StringInterner,
    ) -> impl Iterator<Item = &'a str> + 'a {
        self.properties.keys(interner)
    }

    /// Returns an iterator over (key, value) pairs (excludes id/title/type).
    /// Requires interner to resolve InternedKey → &str.
    #[inline]
    pub fn property_iter<'a>(
        &'a self,
        interner: &'a StringInterner,
    ) -> impl Iterator<Item = (&'a str, &'a Value)> + 'a {
        self.properties.iter(interner)
    }

    /// Returns the number of properties (excludes id/title/type).
    #[inline]
    pub fn property_count(&self) -> usize {
        self.properties.len()
    }

    /// Returns true if the node has the given property key.
    /// Uses hash-based lookup — no interner needed.
    #[inline]
    pub fn has_property(&self, key: &str) -> bool {
        self.properties.contains(InternedKey::from_str(key))
    }

    /// True when properties live in a per-type column store (disk/mapped
    /// backends). For `Columnar`, `property_iter` yields nothing — callers
    /// that need the full property set must consult the type metadata and
    /// read each value through the backend-specific path. For `Map` /
    /// `Compact` (the in-memory backend) `property_iter` already yields
    /// every present property, so the metadata fall-back is unnecessary.
    #[inline]
    pub(crate) fn properties_are_columnar(&self) -> bool {
        matches!(self.properties, PropertyStorage::Columnar { .. })
    }

    /// Clone all properties into a new HashMap<String, Value> (for export/interop).
    /// Requires interner to resolve InternedKey → String.
    #[inline]
    pub fn properties_cloned(&self, interner: &StringInterner) -> HashMap<String, Value> {
        match &self.properties {
            PropertyStorage::Columnar { .. } => {
                self.properties.iter_owned(interner).into_iter().collect()
            }
            _ => self
                .properties
                .iter(interner)
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
        }
    }

    /// Returns the node type as a string reference without allocation.
    /// Requires interner to resolve the InternedKey.
    #[inline]
    pub fn get_node_type_ref<'a>(&self, interner: &'a StringInterner) -> &'a str {
        interner.resolve(self.node_type)
    }

    /// Convert to a NodeInfo snapshot (for Python API / export).
    /// Requires interner to resolve property keys to strings.
    pub fn to_node_info(&self, interner: &StringInterner) -> NodeInfo {
        NodeInfo {
            id: self.id().into_owned(),
            title: self.title().into_owned(),
            node_type: self.node_type_str(interner).to_string(),
            properties: self.properties_cloned(interner),
        }
    }

    /// Insert or update a property, interning the key.
    #[inline]
    pub fn set_property(&mut self, key: &str, value: Value, interner: &mut StringInterner) {
        let interned = interner.get_or_intern(key);
        self.properties.insert(interned, value);
    }

    /// Remove a property by key. Returns the removed value if it existed.
    #[inline]
    pub fn remove_property(&mut self, key: &str) -> Option<Value> {
        self.properties.remove(InternedKey::from_str(key))
    }

    /// Mark a property as cleared. Returns the prior value if it existed.
    ///
    /// Used by the Cypher `REMOVE` clause when the graph is disk-backed.
    /// The disk write-path flush (`flush_node_mut_cache`) only writes
    /// keys *present* in the staged property Map to the column store;
    /// a bare `remove()` of the key from the staged Map leaves the
    /// column store's original value untouched. By inserting
    /// `Value::Null` for the key instead, the flush writes Null to
    /// the column store, and reads return None — matching the
    /// user-facing semantics of REMOVE (the property is no longer
    /// observable on the node).
    ///
    /// For memory/mapped backends, `remove_property` works directly
    /// against the in-place storage and this method isn't needed.
    /// Disk callers detect via `graph.is_disk()` and choose the right
    /// helper.
    #[inline]
    pub fn clear_property(&mut self, key: &str) -> Option<Value> {
        let interned = InternedKey::from_str(key);
        let prior = self.properties.remove(interned);
        self.properties.insert(interned, Value::Null);
        prior
    }
}

pub struct EdgeData {
    pub connection_type: InternedKey,
    pub properties: Vec<(InternedKey, Value)>,
}

// Serialize EdgeData in bincode-compatible struct format:
// connection_type as InternedKey (auto-resolves to string),
// properties as HashMap<InternedKey, Value> (backward-compatible with old format).
impl Serialize for EdgeData {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("EdgeData", 2)?;
        s.serialize_field("connection_type", &self.connection_type)?;
        // Rebuild HashMap for serialization (backward-compatible wire format)
        let props_map: HashMap<&InternedKey, &Value> =
            self.properties.iter().map(|(k, v)| (k, v)).collect();
        s.serialize_field("properties", &props_map)?;
        s.end()
    }
}

// Deserialize EdgeData: read connection_type as InternedKey (from string on disk),
// read properties as HashMap<InternedKey, Value>, convert to Vec.
impl<'de> Deserialize<'de> for EdgeData {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct EdgeDataHelper {
            connection_type: InternedKey,
            #[serde(default)]
            properties: HashMap<InternedKey, Value>,
        }
        let helper = EdgeDataHelper::deserialize(deserializer)?;
        Ok(EdgeData {
            connection_type: helper.connection_type,
            properties: helper.properties.into_iter().collect(),
        })
    }
}

impl std::fmt::Debug for EdgeData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EdgeData")
            .field("connection_type", &self.connection_type)
            .field("properties", &self.properties)
            .finish()
    }
}

impl Clone for EdgeData {
    fn clone(&self) -> Self {
        EdgeData {
            connection_type: self.connection_type,
            properties: self.properties.clone(),
        }
    }
}

impl EdgeData {
    /// Create a new EdgeData, interning connection_type and all property keys.
    pub fn new(
        connection_type: String,
        properties: HashMap<String, Value>,
        interner: &mut StringInterner,
    ) -> Self {
        let ct_key = interner.get_or_intern(&connection_type);
        let interned_props: Vec<(InternedKey, Value)> = properties
            .into_iter()
            .map(|(k, v)| {
                let key = interner.get_or_intern(&k);
                (key, v)
            })
            .collect();
        EdgeData {
            connection_type: ct_key,
            properties: interned_props,
        }
    }

    /// Create EdgeData with pre-interned connection_type and properties.
    pub fn new_interned(
        connection_type: InternedKey,
        properties: Vec<(InternedKey, Value)>,
    ) -> Self {
        EdgeData {
            connection_type,
            properties,
        }
    }

    /// Resolve connection_type to a string via the interner.
    #[inline]
    pub fn connection_type_str<'a>(&self, interner: &'a StringInterner) -> &'a str {
        interner.resolve(self.connection_type)
    }

    /// Returns a reference to an edge property value.
    /// Uses hash-based lookup — no interner needed.
    #[inline]
    pub fn get_property(&self, key: &str) -> Option<&Value> {
        let ik = InternedKey::from_str(key);
        self.properties
            .iter()
            .find(|(k, _)| *k == ik)
            .map(|(_, v)| v)
    }

    /// Returns an iterator over edge property keys.
    /// Requires interner to resolve InternedKey → &str.
    #[inline]
    pub fn property_keys<'a>(
        &'a self,
        interner: &'a StringInterner,
    ) -> impl Iterator<Item = &'a str> {
        self.properties
            .iter()
            .map(move |(k, _)| interner.resolve(*k))
    }

    /// Returns an iterator over (key, value) pairs.
    /// Requires interner to resolve InternedKey → &str.
    #[inline]
    pub fn property_iter<'a>(
        &'a self,
        interner: &'a StringInterner,
    ) -> impl Iterator<Item = (&'a str, &'a Value)> {
        self.properties
            .iter()
            .map(move |(k, v)| (interner.resolve(*k), v))
    }

    /// Returns the number of edge properties.
    #[inline]
    pub fn property_count(&self) -> usize {
        self.properties.len()
    }

    /// Clone all properties into a new HashMap<String, Value> (for export/interop).
    /// Requires interner to resolve InternedKey → String.
    #[inline]
    pub fn properties_cloned(&self, interner: &StringInterner) -> HashMap<String, Value> {
        self.properties
            .iter()
            .map(|(k, v)| (interner.resolve(*k).to_string(), v.clone()))
            .collect()
    }
}

// ============================================================================
// Schema Definition & Validation Types
// ============================================================================

/// Defines the expected schema for a node type
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NodeSchemaDefinition {
    /// Fields that must be present on all nodes of this type
    pub required_fields: Vec<String>,
    /// Fields that may be present (for documentation purposes)
    pub optional_fields: Vec<String>,
    /// Expected types for fields: "string", "integer", "float", "boolean", "datetime"
    pub field_types: HashMap<String, String>,
}

/// Defines the expected schema for a connection type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionSchemaDefinition {
    /// The source node type for this connection
    pub source_type: String,
    /// The target node type for this connection
    pub target_type: String,
    /// Optional cardinality constraint: "one-to-one", "one-to-many", "many-to-one", "many-to-many"
    pub cardinality: Option<String>,
    /// Required properties on the connection
    pub required_properties: Vec<String>,
    /// Expected types for connection properties
    pub property_types: HashMap<String, String>,
}

/// Complete schema definition for the graph
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SchemaDefinition {
    /// Schema definitions for each node type
    pub node_schemas: HashMap<String, NodeSchemaDefinition>,
    /// Schema definitions for each connection type
    pub connection_schemas: HashMap<String, ConnectionSchemaDefinition>,
}

impl SchemaDefinition {
    pub fn new() -> Self {
        SchemaDefinition {
            node_schemas: HashMap::new(),
            connection_schemas: HashMap::new(),
        }
    }

    /// Add a node type schema
    pub fn add_node_schema(&mut self, node_type: String, schema: NodeSchemaDefinition) {
        self.node_schemas.insert(node_type, schema);
    }

    /// Add a connection type schema
    pub fn add_connection_schema(
        &mut self,
        connection_type: String,
        schema: ConnectionSchemaDefinition,
    ) {
        self.connection_schemas.insert(connection_type, schema);
    }
}

/// Represents a validation error found during schema validation
#[derive(Debug, Clone)]
pub enum ValidationError {
    /// A required field is missing from a node
    MissingRequiredField {
        node_type: String,
        node_title: String,
        field: String,
    },
    /// A field has the wrong type
    TypeMismatch {
        node_type: String,
        node_title: String,
        field: String,
        expected_type: String,
        actual_type: String,
    },
    /// A connection has invalid source or target type
    InvalidConnectionEndpoint {
        connection_type: String,
        expected_source: String,
        expected_target: String,
        actual_source: String,
        actual_target: String,
    },
    /// A required property is missing from a connection
    MissingConnectionProperty {
        connection_type: String,
        source_title: String,
        target_title: String,
        property: String,
    },
    /// A node type exists in the graph but not in the schema
    UndefinedNodeType { node_type: String, count: usize },
    /// A connection type exists in the graph but not in the schema
    UndefinedConnectionType {
        connection_type: String,
        count: usize,
    },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::MissingRequiredField {
                node_type,
                node_title,
                field,
            } => {
                write!(
                    f,
                    "Missing required field '{}' on {} node '{}'",
                    field, node_type, node_title
                )
            }
            ValidationError::TypeMismatch {
                node_type,
                node_title,
                field,
                expected_type,
                actual_type,
            } => {
                write!(
                    f,
                    "Type mismatch on {} node '{}': field '{}' expected {}, got {}",
                    node_type, node_title, field, expected_type, actual_type
                )
            }
            ValidationError::InvalidConnectionEndpoint {
                connection_type,
                expected_source,
                expected_target,
                actual_source,
                actual_target,
            } => {
                write!(
                    f,
                    "Invalid connection '{}': expected {}->{}  but found {}->{}",
                    connection_type, expected_source, expected_target, actual_source, actual_target
                )
            }
            ValidationError::MissingConnectionProperty {
                connection_type,
                source_title,
                target_title,
                property,
            } => {
                write!(
                    f,
                    "Missing required property '{}' on {} connection from '{}' to '{}'",
                    property, connection_type, source_title, target_title
                )
            }
            ValidationError::UndefinedNodeType { node_type, count } => {
                write!(
                    f,
                    "Node type '{}' ({} nodes) exists in graph but not defined in schema",
                    node_type, count
                )
            }
            ValidationError::UndefinedConnectionType {
                connection_type,
                count,
            } => {
                write!(f, "Connection type '{}' ({} connections) exists in graph but not defined in schema", connection_type, count)
            }
        }
    }
}

#[cfg(test)]
mod type_id_index_tests {
    use super::*;
    use petgraph::graph::NodeIndex;

    fn integer_index() -> TypeIdIndex {
        let mut m = HashMap::new();
        m.insert(1u32, NodeIndex::new(10));
        m.insert(42u32, NodeIndex::new(20));
        TypeIdIndex::Integer(m)
    }

    #[test]
    fn numeric_coercions_retained() {
        let idx = integer_index();
        // UniqueId / Int64 / Float all hit the integer key.
        assert_eq!(idx.get(&Value::UniqueId(42)), Some(NodeIndex::new(20)));
        assert_eq!(idx.get(&Value::Int64(42)), Some(NodeIndex::new(20)));
        assert_eq!(idx.get(&Value::Float64(42.0)), Some(NodeIndex::new(20)));
        // Non-integral float and out-of-range miss.
        assert_eq!(idx.get(&Value::Float64(42.5)), None);
        assert_eq!(idx.get(&Value::Int64(-1)), None);
    }

    #[test]
    fn string_no_longer_coerces_to_int() {
        // Regression lock (0.10.10): a `String` id must NOT be prefix-stripped
        // into the integer index. `{id:'a1'}` / `{id:'Q1'}` must NOT resolve to
        // `UniqueId(1)` — that was the wrong-node false-positive bug.
        let idx = integer_index();
        assert_eq!(idx.get(&Value::String("a1".into())), None);
        assert_eq!(idx.get(&Value::String("x1".into())), None);
        assert_eq!(idx.get(&Value::String("Q1".into())), None);
        assert_eq!(idx.get(&Value::String("1".into())), None);
    }
}

#[cfg(test)]
mod maintenance_tests {
    use super::*;
    use crate::graph::storage::{GraphRead, GraphWrite};

    /// Helper: create a DirGraph with N Person nodes and edges between consecutive pairs
    fn make_test_graph(num_nodes: usize, num_edges: bool) -> DirGraph {
        let mut g = DirGraph::new();
        for i in 0..num_nodes {
            let mut props = HashMap::new();
            props.insert("age".to_string(), Value::Int64(20 + i as i64));
            let node = NodeData::new(
                Value::UniqueId(i as u32),
                Value::String(format!("Person_{}", i)),
                "Person".to_string(),
                props,
                &mut g.interner,
            );
            let idx = g.graph.add_node(node);
            g.type_indices
                .entry_or_default("Person".to_string())
                .push(idx);
        }
        if num_edges {
            for i in 0..(num_nodes.saturating_sub(1)) {
                let src = NodeIndex::new(i);
                let tgt = NodeIndex::new(i + 1);
                g.graph.add_edge(
                    src,
                    tgt,
                    EdgeData::new("KNOWS".to_string(), HashMap::new(), &mut g.interner),
                );
            }
        }
        g
    }

    #[test]
    fn test_graph_info_clean() {
        let g = make_test_graph(5, true);
        let info = g.graph_info();
        assert_eq!(info.node_count, 5);
        assert_eq!(info.node_capacity, 5);
        assert_eq!(info.node_tombstones, 0);
        assert_eq!(info.edge_count, 4);
        assert_eq!(info.fragmentation_ratio, 0.0);
        assert_eq!(info.type_count, 1);
    }

    #[test]
    fn test_graph_info_after_deletion() {
        let mut g = make_test_graph(5, false);
        // Delete node 2 — leaves a tombstone
        g.graph.remove_node(NodeIndex::new(2));
        let info = g.graph_info();
        assert_eq!(info.node_count, 4);
        assert_eq!(info.node_capacity, 5); // Still 5 slots
        assert_eq!(info.node_tombstones, 1);
        assert!(info.fragmentation_ratio > 0.19 && info.fragmentation_ratio < 0.21);
    }

    #[test]
    fn test_graph_info_empty() {
        let g = DirGraph::new();
        let info = g.graph_info();
        assert_eq!(info.node_count, 0);
        assert_eq!(info.node_capacity, 0);
        assert_eq!(info.fragmentation_ratio, 0.0);
    }

    #[test]
    fn test_reindex_rebuilds_type_indices() {
        let mut g = make_test_graph(5, false);

        // Manually corrupt type_indices (simulate drift)
        g.type_indices.clear();
        assert!(g.type_indices.is_empty());

        g.reindex();

        // type_indices should be rebuilt
        assert_eq!(g.type_indices.len(), 1);
        assert_eq!(g.type_indices.get("Person").unwrap().len(), 5);
    }

    #[test]
    fn test_reindex_rebuilds_property_indices() {
        let mut g = make_test_graph(5, false);

        // Create a property index
        g.create_index("Person", "age");
        assert!(g.has_index("Person", "age"));

        // Manually corrupt the property index
        g.property_indices
            .get_mut(&("Person".to_string(), "age".to_string()))
            .unwrap()
            .clear();

        g.reindex();

        // Property index should be rebuilt with correct data
        let stats = g.get_index_stats("Person", "age").unwrap();
        assert_eq!(stats.unique_values, 5); // ages 20..24
        assert_eq!(stats.total_entries, 5);
    }

    #[test]
    fn test_reindex_rebuilds_composite_indices() {
        let mut g = make_test_graph(5, false);
        g.create_composite_index("Person", &["age"]);
        assert!(g.has_composite_index("Person", &["age".to_string()]));

        // Corrupt composite index
        g.composite_indices.values_mut().for_each(|v| v.clear());

        g.reindex();

        let stats = g
            .get_composite_index_stats("Person", &["age".to_string()])
            .unwrap();
        assert_eq!(stats.unique_values, 5);
    }

    #[test]
    fn test_reindex_clears_id_indices() {
        let mut g = make_test_graph(3, false);
        g.build_id_index("Person");
        assert!(g.id_indices.contains_key("Person"));

        g.reindex();

        // id_indices should be cleared (lazy rebuild on next access)
        assert!(g.id_indices.is_empty());
    }

    #[test]
    fn test_reindex_after_deletion() {
        let mut g = make_test_graph(5, false);
        // Delete node 2
        g.graph.remove_node(NodeIndex::new(2));
        // type_indices still has the stale entry
        assert_eq!(g.type_indices.get("Person").unwrap().len(), 5);

        g.reindex();

        // Now type_indices should reflect only 4 live nodes
        assert_eq!(g.type_indices.get("Person").unwrap().len(), 4);
        // And none of them should be index 2
        assert!(!g
            .type_indices
            .get("Person")
            .unwrap()
            .contains(&NodeIndex::new(2)));
    }

    #[test]
    fn test_vacuum_noop_when_clean() {
        let mut g = make_test_graph(5, true);
        let mapping = g.vacuum();
        assert!(mapping.is_empty()); // No remapping needed
        assert_eq!(g.graph.node_count(), 5);
        assert_eq!(g.graph_info().node_tombstones, 0);
    }

    #[test]
    fn test_vacuum_compacts_after_deletion() {
        let mut g = make_test_graph(5, true);
        // Delete middle node (creates tombstone)
        g.graph.remove_node(NodeIndex::new(2));
        assert_eq!(g.graph.node_count(), 4);
        assert_eq!(g.graph_info().node_tombstones, 1);

        let mapping = g.vacuum();

        // After vacuum: no tombstones, indices are contiguous
        assert_eq!(g.graph.node_count(), 4);
        assert_eq!(g.graph_info().node_tombstones, 0);
        assert_eq!(g.graph_info().node_capacity, 4);

        // Mapping should have 4 entries (one for each surviving node)
        assert_eq!(mapping.len(), 4);
    }

    #[test]
    fn test_vacuum_preserves_node_data() {
        let mut g = make_test_graph(3, false);
        g.graph.remove_node(NodeIndex::new(1)); // Delete Person_1

        let mapping = g.vacuum();

        // Verify all surviving nodes are present with correct data
        let mut titles: Vec<String> = Vec::new();
        for idx in g.graph.node_indices() {
            if let Some(node) = g.graph.node_weight(idx) {
                if let Value::String(s) = &*node.title() {
                    titles.push(s.clone());
                }
            }
        }
        titles.sort();
        assert_eq!(titles, vec!["Person_0", "Person_2"]);
        assert_eq!(mapping.len(), 2);
    }

    #[test]
    fn test_vacuum_preserves_edges() {
        let mut g = make_test_graph(4, true);
        // Edges: 0→1, 1→2, 2→3
        // Delete node 0 (and its edge to 1)
        g.graph.remove_node(NodeIndex::new(0));
        // Remaining edges should be 1→2, 2→3

        let _mapping = g.vacuum();

        assert_eq!(g.graph.edge_count(), 2);
        assert_eq!(g.graph.node_count(), 3);
    }

    #[test]
    fn test_vacuum_rebuilds_type_indices() {
        let mut g = make_test_graph(5, false);
        g.graph.remove_node(NodeIndex::new(2));

        g.vacuum();

        // type_indices should point to valid, contiguous indices
        assert_eq!(g.type_indices.get("Person").unwrap().len(), 4);
        for idx in g.type_indices.get("Person").unwrap().iter() {
            assert!(g.graph.node_weight(idx).is_some());
        }
    }

    #[test]
    fn test_vacuum_rebuilds_property_indices() {
        let mut g = make_test_graph(5, false);
        g.create_index("Person", "age");
        g.graph.remove_node(NodeIndex::new(2));

        g.vacuum();

        // Property index should still exist with correct entries
        assert!(g.has_index("Person", "age"));
        let stats = g.get_index_stats("Person", "age").unwrap();
        assert_eq!(stats.total_entries, 4); // 5 - 1 deleted
    }

    #[test]
    fn test_vacuum_heavy_fragmentation() {
        let mut g = make_test_graph(100, false);
        // Delete every other node — 50% fragmentation
        for i in (0..100).step_by(2) {
            g.graph.remove_node(NodeIndex::new(i));
        }
        assert_eq!(g.graph.node_count(), 50);
        let info = g.graph_info();
        assert!(info.fragmentation_ratio > 0.49);

        let mapping = g.vacuum();

        assert_eq!(mapping.len(), 50);
        assert_eq!(g.graph.node_count(), 50);
        assert_eq!(g.graph_info().node_tombstones, 0);
        assert_eq!(g.graph_info().fragmentation_ratio, 0.0);
    }

    // ========================================================================
    // Incremental Index Update Tests
    // ========================================================================

    #[test]
    fn test_update_property_indices_for_add() {
        let mut g = DirGraph::new();
        // Add a node and create an index
        let mut props = HashMap::new();
        props.insert("city".to_string(), Value::String("Oslo".to_string()));
        let n0 = g.graph.add_node(NodeData::new(
            Value::Int64(1),
            Value::String("Alice".to_string()),
            "Person".to_string(),
            props,
            &mut g.interner,
        ));
        g.type_indices
            .entry_or_default("Person".to_string())
            .push(n0);
        g.create_index("Person", "city");

        // Add a second node and call the helper
        let mut props2 = HashMap::new();
        props2.insert("city".to_string(), Value::String("Bergen".to_string()));
        let n1 = g.graph.add_node(NodeData::new(
            Value::Int64(2),
            Value::String("Bob".to_string()),
            "Person".to_string(),
            props2,
            &mut g.interner,
        ));
        g.type_indices
            .entry_or_default("Person".to_string())
            .push(n1);
        g.update_property_indices_for_add("Person", n1);

        // Verify index was updated
        let oslo = g.lookup_by_index("Person", "city", &Value::String("Oslo".to_string()));
        assert_eq!(oslo.unwrap().len(), 1);
        let bergen = g.lookup_by_index("Person", "city", &Value::String("Bergen".to_string()));
        let bergen = bergen.unwrap();
        assert_eq!(bergen.len(), 1);
        assert_eq!(bergen[0], n1);
    }

    #[test]
    fn test_update_property_indices_for_set() {
        let mut g = DirGraph::new();
        let mut props = HashMap::new();
        props.insert("city".to_string(), Value::String("Oslo".to_string()));
        let n0 = g.graph.add_node(NodeData::new(
            Value::Int64(1),
            Value::String("Alice".to_string()),
            "Person".to_string(),
            props,
            &mut g.interner,
        ));
        g.type_indices
            .entry_or_default("Person".to_string())
            .push(n0);
        g.create_index("Person", "city");

        // Simulate SET n.city = 'Bergen'
        let old_val = Value::String("Oslo".to_string());
        let new_val = Value::String("Bergen".to_string());
        // Actually change the property on the node
        if let Some(node) = g.graph.node_weight_mut(n0) {
            node.set_property("city", new_val.clone(), &mut g.interner);
        }
        g.update_property_indices_for_set("Person", n0, "city", Some(&old_val), &new_val);

        // Verify: Oslo bucket should be empty, Bergen should have the node
        let oslo = g.lookup_by_index("Person", "city", &Value::String("Oslo".to_string()));
        assert!(oslo.is_none() || oslo.unwrap().is_empty());
        let bergen = g.lookup_by_index("Person", "city", &Value::String("Bergen".to_string()));
        assert_eq!(bergen.unwrap(), vec![n0]);
    }

    #[test]
    fn test_update_property_indices_for_remove() {
        let mut g = DirGraph::new();
        let mut props = HashMap::new();
        props.insert("city".to_string(), Value::String("Oslo".to_string()));
        let n0 = g.graph.add_node(NodeData::new(
            Value::Int64(1),
            Value::String("Alice".to_string()),
            "Person".to_string(),
            props,
            &mut g.interner,
        ));
        g.type_indices
            .entry_or_default("Person".to_string())
            .push(n0);
        g.create_index("Person", "city");

        // Simulate REMOVE n.city
        let old_val = Value::String("Oslo".to_string());
        if let Some(node) = g.graph.node_weight_mut(n0) {
            node.remove_property("city");
        }
        g.update_property_indices_for_remove("Person", n0, "city", &old_val);

        // Verify: Oslo bucket should be empty
        let oslo = g.lookup_by_index("Person", "city", &Value::String("Oslo".to_string()));
        assert!(oslo.is_none() || oslo.unwrap().is_empty());
    }

    #[test]
    fn test_update_composite_index_on_property_change() {
        let mut g = DirGraph::new();
        let mut props = HashMap::new();
        props.insert("city".to_string(), Value::String("Oslo".to_string()));
        props.insert("age".to_string(), Value::Int64(30));
        let n0 = g.graph.add_node(NodeData::new(
            Value::Int64(1),
            Value::String("Alice".to_string()),
            "Person".to_string(),
            props,
            &mut g.interner,
        ));
        g.type_indices
            .entry_or_default("Person".to_string())
            .push(n0);
        g.create_composite_index("Person", &["city", "age"]);

        // Verify initial state
        let key = (
            "Person".to_string(),
            vec!["city".to_string(), "age".to_string()],
        );
        assert!(g.composite_indices.get(&key).unwrap().len() == 1);

        // Change city to Bergen
        let old_val = Value::String("Oslo".to_string());
        let new_val = Value::String("Bergen".to_string());
        if let Some(node) = g.graph.node_weight_mut(n0) {
            node.set_property("city", new_val.clone(), &mut g.interner);
        }
        g.update_property_indices_for_set("Person", n0, "city", Some(&old_val), &new_val);

        // Verify: old composite value gone, new one present
        let comp_map = g.composite_indices.get(&key).unwrap();
        let old_comp = CompositeValue(vec![Value::String("Oslo".to_string()), Value::Int64(30)]);
        let new_comp = CompositeValue(vec![Value::String("Bergen".to_string()), Value::Int64(30)]);
        assert!(!comp_map.contains_key(&old_comp) || comp_map.get(&old_comp).unwrap().is_empty());
        assert_eq!(comp_map.get(&new_comp).unwrap(), &vec![n0]);
    }

    #[test]
    fn test_no_update_when_no_index_exists() {
        let mut g = DirGraph::new();
        let mut props = HashMap::new();
        props.insert("city".to_string(), Value::String("Oslo".to_string()));
        let n0 = g.graph.add_node(NodeData::new(
            Value::Int64(1),
            Value::String("Alice".to_string()),
            "Person".to_string(),
            props,
            &mut g.interner,
        ));
        g.type_indices
            .entry_or_default("Person".to_string())
            .push(n0);
        // No index created — these should be no-ops without crash
        g.update_property_indices_for_add("Person", n0);
        g.update_property_indices_for_set(
            "Person",
            n0,
            "city",
            Some(&Value::String("Oslo".to_string())),
            &Value::String("Bergen".to_string()),
        );
        g.update_property_indices_for_remove(
            "Person",
            n0,
            "city",
            &Value::String("Oslo".to_string()),
        );
        assert!(g.property_indices.is_empty());
    }

    // ─── Columnar storage tests ──────────────────────────────────────────

    #[test]
    fn test_enable_columnar_preserves_properties() {
        let mut g = make_test_graph(5, false);
        // Add metadata so columnar knows types
        let mut meta = HashMap::new();
        meta.insert("age".to_string(), "int64".to_string());
        g.node_type_metadata.insert("Person".to_string(), meta);
        g.compact_properties();

        // Snapshot properties before
        let before: Vec<(Value, Value, i64)> = g
            .type_indices
            .get("Person")
            .unwrap()
            .iter()
            .map(|idx| {
                let n = g.graph.node_weight(idx).unwrap();
                let age = n
                    .get_property("age")
                    .map(|c| match c.as_ref() {
                        Value::Int64(v) => *v,
                        _ => panic!("expected Int64"),
                    })
                    .unwrap();
                (n.id().into_owned(), n.title().into_owned(), age)
            })
            .collect();

        g.enable_columnar();
        assert!(g.is_columnar());

        // Verify properties match
        let after: Vec<(Value, Value, i64)> = g
            .type_indices
            .get("Person")
            .unwrap()
            .iter()
            .map(|idx| {
                let n = g.graph.node_weight(idx).unwrap();
                let age = n
                    .get_property("age")
                    .map(|c| match c.as_ref() {
                        Value::Int64(v) => *v,
                        _ => panic!("expected Int64"),
                    })
                    .unwrap();
                (n.id().into_owned(), n.title().into_owned(), age)
            })
            .collect();

        assert_eq!(before, after);
    }

    #[test]
    fn test_columnar_roundtrip_via_disable() {
        let mut g = make_test_graph(3, false);
        let mut meta = HashMap::new();
        meta.insert("age".to_string(), "int64".to_string());
        g.node_type_metadata.insert("Person".to_string(), meta);
        g.compact_properties();

        // Enable columnar, then disable back to Compact
        g.enable_columnar();
        assert!(g.is_columnar());
        g.disable_columnar();
        assert!(!g.is_columnar());

        // Verify properties still work
        let idx = g.type_indices.get("Person").unwrap().get(0).unwrap();
        let node = g.graph.node_weight(idx).unwrap();
        assert!(matches!(node.properties, PropertyStorage::Compact { .. }));
        assert!(node.get_property("age").is_some());
    }

    #[test]
    fn test_columnar_set_property() {
        let mut g = make_test_graph(2, false);
        let mut meta = HashMap::new();
        meta.insert("age".to_string(), "int64".to_string());
        g.node_type_metadata.insert("Person".to_string(), meta);
        g.compact_properties();
        g.enable_columnar();

        let idx = g.type_indices.get("Person").unwrap().get(0).unwrap();
        let node = g.graph.node_weight_mut(idx).unwrap();

        // Update existing property
        node.set_property("age", Value::Int64(99), &mut g.interner);
        assert_eq!(
            node.get_property("age").map(|c| c.into_owned()),
            Some(Value::Int64(99))
        );
    }

    #[test]
    fn test_columnar_property_count_and_keys() {
        let mut g = make_test_graph(2, false);
        let mut meta = HashMap::new();
        meta.insert("age".to_string(), "int64".to_string());
        g.node_type_metadata.insert("Person".to_string(), meta);
        g.compact_properties();
        g.enable_columnar();

        let idx = g.type_indices.get("Person").unwrap().get(0).unwrap();
        let node = g.graph.node_weight(idx).unwrap();

        assert_eq!(node.property_count(), 1); // just "age"
        let keys: Vec<&str> = node.property_keys(&g.interner).collect();
        assert_eq!(keys, vec!["age"]);
    }

    #[test]
    fn test_columnar_serialize_roundtrip() {
        let mut g = make_test_graph(3, false);
        let mut meta = HashMap::new();
        meta.insert("age".to_string(), "int64".to_string());
        g.node_type_metadata.insert("Person".to_string(), meta);
        g.compact_properties();
        g.enable_columnar();

        // Serialize (Columnar should produce same output as Compact)
        let serialized = {
            let _guard = SerdeSerializeGuard::new(&g.interner);
            bincode::serialize(&g.graph).unwrap()
        };

        // Deserialize into a new graph — will come back as Map
        let graph2: GraphBackend = {
            let _guard = SerdeDeserializeGuard::new(&mut g.interner);
            bincode::deserialize(&serialized).unwrap()
        };
        let node0 = graph2.node_weight(NodeIndex::new(0)).unwrap();

        // Properties should survive the round-trip
        assert!(node0.get_property("age").is_some());
    }
}

#[cfg(test)]
mod embedding_store_tests {
    use super::*;

    #[test]
    fn text_hash_is_deterministic_and_distinguishing() {
        assert_eq!(
            EmbeddingStore::text_hash("hello"),
            EmbeddingStore::text_hash("hello"),
            "same text must hash identically (cross-process stable)"
        );
        assert_ne!(
            EmbeddingStore::text_hash("hello"),
            EmbeddingStore::text_hash("world"),
        );
        assert_ne!(
            EmbeddingStore::text_hash("hello"),
            EmbeddingStore::text_hash("Hello"),
        );
    }

    #[test]
    fn is_stale_covers_missing_changed_and_unhashed() {
        let mut store = EmbeddingStore::new(2);
        let h = EmbeddingStore::text_hash("v1");

        // No embedding yet → stale.
        assert!(store.is_stale(7, h));

        // Embedding present + matching hash → not stale.
        store.set_embedding(7, &[1.0, 2.0]);
        store.set_text_hash(7, h);
        assert!(!store.is_stale(7, h));

        // Text changed (different hash) → stale.
        assert!(store.is_stale(7, EmbeddingStore::text_hash("v2")));

        // Embedding present but no recorded hash (e.g. add_embeddings) → stale,
        // so mode='changed' will (re)hash it on the next pass.
        store.set_embedding(9, &[3.0, 4.0]);
        assert!(store.is_stale(9, EmbeddingStore::text_hash("anything")));
    }

    #[test]
    fn new_store_has_empty_provenance() {
        let store = EmbeddingStore::new(4);
        assert_eq!(store.model_id, None);
        assert!(store.text_hashes.is_empty());
    }
}
