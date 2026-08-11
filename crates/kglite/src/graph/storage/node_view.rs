//! [`NodeView`] — the authoritative read route for a node's properties.
//!
//! # Why this exists
//!
//! A columnar node's properties live in a per-type
//! [`ColumnStore`](crate::graph::storage::column_store::ColumnStore) that the
//! **storage backend owns**. Before D1 three `Arc`s pointed at the same store —
//! the node's own handle, a `DirGraph`-level map, and `DiskGraph`'s — so a read
//! resolved *one particular replica* rather than the owner, which is how two
//! shipped defects arose (an empty columnar `property_iter`, and a spill that
//! reclaimed nothing). There is one owner now, and this type is how a caller
//! reaches it.
//!
//! `NodeView` is the single place a node's property read resolves its store.
//! Callers ask the storage backend for a view
//! ([`GraphRead::node_view`](crate::graph::storage::GraphRead::node_view)) and
//! then read through it. When the backend becomes the sole owner of the stores,
//! only [`NodeView::from_node_data`] and the backend accessors change; every
//! caller keeps compiling and keeps its meaning.
//!
//! # The columnar completeness contract
//!
//! `NodeData::property_iter` yields **nothing** for
//! `PropertyStorage::Columnar` — it cannot, because columnar values are
//! constructed on read and there is no `&'a Value` to hand out. Every
//! enumeration method on `NodeView` ([`NodeView::property_pairs`],
//! [`NodeView::property_keys`], [`NodeView::properties_cloned`],
//! [`NodeView::property_pairs_named`]) is **complete for every storage
//! variant**, columnar included. That is the contract: if you enumerate through
//! a `NodeView` you see the node's real properties.

use std::borrow::Cow;
use std::collections::HashMap;

use crate::datatypes::Value;
use crate::graph::schema::{InternedKey, NodeData, NodeInfo, PropertyStorage, StringInterner};
use crate::graph::storage::column_store::ColumnStore;

/// A borrowed, backend-resolved read handle for one node.
///
/// Cheap to copy (two words + a row id). Obtained from
/// [`GraphRead::node_view`](crate::graph::storage::GraphRead::node_view).
///
/// # Lifetime discipline
///
/// A `NodeView` borrows the storage backend. On the disk backend the borrowed
/// `NodeData` lives in the per-query arena, so a view must not outlive the
/// `begin_query()` guard, and must never be held across a `Python::attach`
/// boundary or a GIL release — resolve to owned [`Value`]s at such a boundary
/// (that is what [`NodeView::properties_cloned`] and
/// [`NodeView::to_node_info`] are for).
#[derive(Clone, Copy)]
pub struct NodeView<'a> {
    data: &'a NodeData,
    /// The resolved column store for this node's row, when the node's
    /// properties are columnar. Resolving once per view rather than once per
    /// read is the point of the type.
    store: Option<(&'a ColumnStore, u32)>,
}

impl<'a> NodeView<'a> {
    /// Pair a node with the store its backend owns.
    ///
    /// The only constructor. `store` comes from
    /// [`GraphRead::column_store`](crate::graph::storage::GraphRead::column_store),
    /// resolved by the node's type — a node itself knows only its `row_id`
    /// (D1 Phase 3). A `None` store on a columnar node means the backend has
    /// no store for that type, which reads as an empty property set.
    #[inline]
    pub(crate) fn new(data: &'a NodeData, store: Option<(&'a ColumnStore, u32)>) -> Self {
        NodeView { data, store }
    }

    /// Escape hatch to the underlying `NodeData`.
    ///
    /// Only for callers that need identity/whole-struct semantics (clone,
    /// equality, serialization) rather than a property read. **Not** for
    /// property reads — those must go through this type's methods.
    #[inline]
    pub fn data(&self) -> &'a NodeData {
        self.data
    }

    /// The node's primary type key.
    #[inline]
    pub fn node_type(&self) -> InternedKey {
        self.data.node_type
    }

    /// The node's primary type, resolved to a string.
    #[inline]
    pub fn node_type_str<'i>(&self, interner: &'i StringInterner) -> &'i str {
        interner.resolve(self.data.node_type)
    }

    /// The node's primary type, resolved to a string. Alias of
    /// [`NodeView::node_type_str`], matching `NodeData`'s two spellings so
    /// migrated call sites read unchanged.
    #[inline]
    pub fn get_node_type_ref<'i>(&self, interner: &'i StringInterner) -> &'i str {
        interner.resolve(self.data.node_type)
    }

    /// The node's id (resolving the mapped-mode `Null` sentinel through the
    /// column store).
    #[inline]
    pub fn id(&self) -> Cow<'a, Value> {
        if matches!(self.data.id, Value::Null) {
            if let Some((store, row_id)) = self.store {
                if let Some(v) = store.get_id(row_id) {
                    return Cow::Owned(v);
                }
            }
        }
        Cow::Borrowed(&self.data.id)
    }

    /// The node's title (resolving the mapped-mode `Null` sentinel through the
    /// column store).
    #[inline]
    pub fn title(&self) -> Cow<'a, Value> {
        if matches!(self.data.title, Value::Null) {
            if let Some((store, row_id)) = self.store {
                if let Some(v) = store.get_title(row_id) {
                    return Cow::Owned(v);
                }
            }
        }
        Cow::Borrowed(&self.data.title)
    }

    /// Read a property by interned key. `None` when absent or `Value::Null`.
    #[inline]
    pub fn get(&self, key: InternedKey) -> Option<Cow<'a, Value>> {
        match self.store {
            Some((store, row_id)) => store.get(row_id, key).map(Cow::Owned),
            None => self.data.properties.get(key),
        }
    }

    /// Read a property by interned key, owned. Cheaper than [`NodeView::get`]
    /// for callers that always need ownership.
    #[inline]
    pub fn get_value(&self, key: InternedKey) -> Option<Value> {
        match self.store {
            Some((store, row_id)) => store.get(row_id, key),
            None => self.data.properties.get_value(key),
        }
    }

    /// Read a property by name (excludes `id` / `title`).
    #[inline]
    pub fn get_property(&self, key: &str) -> Option<Cow<'a, Value>> {
        self.get(InternedKey::from_str(key))
    }

    /// Read a property by name, owned (excludes `id` / `title`).
    #[inline]
    pub fn get_property_value(&self, key: &str) -> Option<Value> {
        self.get_value(InternedKey::from_str(key))
    }

    /// Read a *field* by name — `id` and `title` resolve to the node's
    /// identity columns, anything else to a property.
    #[inline]
    pub fn get_field_ref(&self, field: &str) -> Option<Cow<'a, Value>> {
        match field {
            "id" => Some(self.id()),
            "title" => Some(self.title()),
            _ => self.get(InternedKey::from_str(field)),
        }
    }

    /// `true` when the property is present and non-`Null`.
    #[inline]
    pub fn contains(&self, key: InternedKey) -> bool {
        match self.store {
            Some((store, row_id)) => store.get(row_id, key).is_some(),
            None => self.data.properties.contains(key),
        }
    }

    /// `true` when the named property is present and non-`Null`.
    #[inline]
    pub fn has_property(&self, key: &str) -> bool {
        self.contains(InternedKey::from_str(key))
    }

    /// `true` when this node's properties live in a per-type column store.
    ///
    /// Callers should not need this — every enumeration method on `NodeView`
    /// is already complete for columnar storage. It survives for the few sites
    /// that branch on storage shape for *schema* reasons (e.g. completing a
    /// projection from type metadata).
    #[inline]
    pub fn properties_are_columnar(&self) -> bool {
        self.store.is_some()
    }

    /// Number of present (non-`Null`) properties.
    #[inline]
    pub fn property_count(&self) -> usize {
        match self.store {
            Some((store, row_id)) => store.row_properties(row_id).len(),
            None => self.data.properties.len(),
        }
    }

    /// Allocation-free string equality against a property.
    ///
    /// `None` — absent/null; `Some(true)` — equal; `Some(false)` — present but
    /// different (including non-string values).
    #[inline]
    pub fn str_prop_eq(&self, key: InternedKey, target: &str) -> Option<bool> {
        match self.store {
            Some((store, row_id)) => store.str_prop_eq(row_id, key, target),
            None => self.data.properties.str_prop_eq(key, target),
        }
    }

    /// Case-insensitive substring test on a string-typed field. `false` when
    /// the field is missing or non-string; `needle_lower` must already be
    /// lowercased.
    pub fn field_contains_ci(&self, field: &str, needle_lower: &str) -> bool {
        self.get_field_ref(field)
            .and_then(|v| match &*v {
                Value::String(s) => Some(s.to_lowercase().contains(needle_lower)),
                _ => None,
            })
            .unwrap_or(false)
    }

    /// Case-insensitive prefix test on a string-typed field. `false` when the
    /// field is missing or non-string; `prefix_lower` must already be
    /// lowercased.
    pub fn field_starts_with_ci(&self, field: &str, prefix_lower: &str) -> bool {
        self.get_field_ref(field)
            .and_then(|v| match &*v {
                Value::String(s) => Some(s.to_lowercase().starts_with(prefix_lower)),
                _ => None,
            })
            .unwrap_or(false)
    }

    /// Every present property as `(interned key, owned value)`.
    ///
    /// **Complete for columnar storage** — the removed
    /// `NodeData::property_iter` yielded nothing there.
    pub fn property_pairs(&self) -> Vec<(InternedKey, Value)> {
        match self.store {
            Some((store, row_id)) => store.row_properties(row_id),
            None => match &self.data.properties {
                // `Map` deliberately keeps `Value::Null` entries visible —
                // `NodeData::clear_property` stages a REMOVE that way for the
                // disk flush. Filtering them here would change what every
                // existing enumeration caller sees.
                PropertyStorage::Map(map) => map.iter().map(|(k, v)| (*k, v.clone())).collect(),
                PropertyStorage::Compact { schema, values } => schema
                    .slots
                    .iter()
                    .enumerate()
                    .filter_map(|(i, ik)| {
                        values.get(i).and_then(|v| {
                            if matches!(v, Value::Null) {
                                None
                            } else {
                                Some((*ik, v.clone()))
                            }
                        })
                    })
                    .collect(),
                PropertyStorage::Columnar(_) => unreachable!("store resolved above"),
            },
        }
    }

    /// Every present property key, resolved to a string.
    ///
    /// **Complete for columnar storage.** Keys that the interner cannot resolve
    /// are skipped, matching the pre-existing `PropertyStorage::keys` contract.
    pub fn property_keys(&self, interner: &'a StringInterner) -> Vec<&'a str> {
        match self.store {
            Some((store, row_id)) => store
                .row_properties(row_id)
                .into_iter()
                .filter_map(|(ik, _)| interner.try_resolve(ik))
                .collect(),
            None => self.data.properties.keys(interner).collect(),
        }
    }

    /// Every present property as `(name, owned value)`.
    ///
    /// **Complete for columnar storage** — the replacement for
    /// `property_iter().map(|(k, v)| (k.to_string(), v.clone()))`, which
    /// silently produced an empty vector for saved graphs.
    pub fn property_pairs_named(&self, interner: &StringInterner) -> Vec<(String, Value)> {
        self.property_pairs()
            .into_iter()
            .filter_map(|(ik, v)| interner.try_resolve(ik).map(|s| (s.to_string(), v)))
            .collect()
    }

    /// Every present property as a `HashMap<String, Value>` (export / interop).
    ///
    /// **Complete for columnar storage.**
    #[inline]
    pub fn properties_cloned(&self, interner: &StringInterner) -> HashMap<String, Value> {
        self.property_pairs_named(interner).into_iter().collect()
    }

    /// Owned snapshot of the whole node (Python API / export).
    #[inline]
    pub fn to_node_info(&self, interner: &StringInterner) -> NodeInfo {
        NodeInfo {
            id: self.id().into_owned(),
            title: self.title().into_owned(),
            node_type: self.node_type_str(interner).to_string(),
            properties: self.properties_cloned(interner),
        }
    }
}

impl std::fmt::Debug for NodeView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeView")
            .field("id", &self.data.id)
            .field("title", &self.data.title)
            .field("node_type", &self.data.node_type)
            .field("columnar_row", &self.store.map(|(_, r)| r))
            .finish()
    }
}
