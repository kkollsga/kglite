//! [`PropMap`] — the property container behind `NodeValue`, `RelValue` and
//! `Value::Map`.
//!
//! # What this is
//!
//! An `Arc`'d, **sorted flat** map: `Arc<Vec<(PropKey, Value)>>`, kept in key
//! order and binary-searched. Two properties follow from that shape:
//!
//! - **Cloning a materialised node's properties is a refcount bump**, not a
//!   deep copy of the map. Every `collect(n)`, `ORDER BY n`, `WITH`-chain hop
//!   and row-to-row copy pays one atomic instead of one allocation per
//!   property.
//! - **One contiguous allocation** replaces a `BTreeMap`'s node, so iteration,
//!   comparison and serialization walk memory linearly.
//!
//! Key order is byte-for-byte `String`'s order, so `Ord`, `Eq`, iteration and
//! the serialized form all stay exactly where `BTreeMap<String, Value>` left
//! them.
//!
//! # Keys are owned, and that is a measured decision
//!
//! [`PropKey`] is an owned `String`. The N0 spike
//! (`dev-docs/bench/results/2026-08-19-nodevalue-spike.md`) predicted that
//! **sharing** keys — handing every row the interner's own `Arc<str>` — would
//! combine with this container for a +7.2–9.0% win on `return_node_10k`.
//! Built faithfully and measured, it does the opposite:
//!
//! | key type | `return_node_10k` | `collect_n_10k` | `properties_n_10k` |
//! |---|---|---|---|
//! | `Arc<str>` shared from the interner | **−15…−19%** | +16% | −24% |
//! | owned `String` (this) | +1.7% | +7.8% | −1.4% |
//! | owned `Box<str>` | +0.9% | +6.3% | −1.9% |
//!
//! Reproducible across three independent release runs, with the unchanged-path
//! controls flat. The mechanism: a shared key costs an **atomic refcount pair**
//! (one increment at construction, one decrement at drop) per property per
//! row, and on this platform that is dearer than the short-string allocation it
//! was meant to replace — 7 keys × 10k rows is 140k atomic RMWs on a handful of
//! cache lines. It is not the thread-local interner's hash lookups (the spike's
//! stated suspicion): this implementation rides the interner resolve that
//! `absorb_stored` already performs and adds no lookup at all, and still loses.
//!
//! So the container ships and key sharing does not. If key sharing is revisited,
//! the shape that could still pay is a `u32` index into one `Arc`'d per-graph
//! key table — one refcount for the whole map rather than one per key — which
//! is a different design, not a tweak to this one.
//!
//! # Byte invisibility is the hard constraint
//!
//! `.kgl` snapshots, WAL frames and CDC payloads carry these maps. The custom
//! [`Serialize`]/[`Deserialize`] below emit and accept **exactly** postcard's
//! map framing — the same bytes `BTreeMap<String, Value>` produced. The
//! instrument that proves it is
//! `crates/kglite/src/graph/value_byte_identity_tests.rs`, whose pinned hex
//! literals and `.kgl` digest this change passes **unchanged**. A red line
//! there means a format change, which is a deliberate decision with a magic
//! bump, never a refreshed constant.
//!
//! # Ordering is a user-visible contract
//!
//! `NodeValue`/`RelValue` derive `Ord`, so `ORDER BY n` reaches `properties` as
//! its final tie-break. `Vec<(PropKey, Value)>` compares lexicographically over
//! `(key, value)` pairs exactly as `BTreeMap`'s `Ord` does (which is defined as
//! `self.iter().cmp(other.iter())`). The orderings are pinned in
//! `crate::datatypes::value_shape_tests`, and `PropMap`'s own unit tests assert
//! it against a `BTreeMap` computing the same comparisons.

use serde::de::{MapAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use super::values::Value;

/// A property name inside a [`PropMap`].
///
/// Owned rather than shared — see the module docs for the measurement that
/// settled it. Kept as an alias so the choice has one place to change.
pub type PropKey = String;

/// An `Arc`'d, sorted, flat property map — see the module docs.
///
/// The invariant every constructor upholds: **the backing vector is sorted by
/// key and carries no duplicate keys.** All reads (`get`, `contains_key`) rely
/// on it for binary search, and `Eq`/`Ord`/`Hash`/`Serialize` rely on it for
/// determinism.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PropMap(Arc<Vec<(PropKey, Value)>>);

impl PropMap {
    /// The empty map. Does not allocate a backing vector's buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from pairs that are **already sorted by key and de-duplicated**.
    ///
    /// The hot construction path (`collect_node_properties`) sorts once at the
    /// end of its walk and calls this. Debug builds assert the invariant, so a
    /// caller that gets it wrong fails loudly in the profile the test suite
    /// runs under rather than silently breaking binary search.
    pub fn from_sorted_pairs(pairs: Vec<(PropKey, Value)>) -> Self {
        debug_assert!(
            pairs.windows(2).all(|w| *w[0].0 < *w[1].0),
            "PropMap::from_sorted_pairs got unsorted or duplicated keys"
        );
        Self(Arc::new(pairs))
    }

    /// Build from arbitrary pairs: sorts by key and keeps the **last** value
    /// written for a duplicated key, matching `BTreeMap::insert` semantics for
    /// a caller that pushed the same key twice.
    ///
    /// The sort is stable, so "last wins" is the last *push*, not an arbitrary
    /// one of the duplicates.
    pub fn from_pairs(mut pairs: Vec<(PropKey, Value)>) -> Self {
        pairs.sort_by(|a, b| a.0.cmp(&b.0));
        dedup_keep_last(&mut pairs);
        Self(Arc::new(pairs))
    }

    /// Value for `key`, or `None`. O(log n) binary search over one contiguous
    /// allocation.
    #[inline]
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.position(key).map(|i| &self.0[i].1)
    }

    /// The stored key and its value — for callers that want to reuse the
    /// shared key rather than allocate their own copy of the name.
    #[inline]
    pub fn get_key_value(&self, key: &str) -> Option<(&PropKey, &Value)> {
        self.position(key).map(|i| {
            let (k, v) = &self.0[i];
            (k, v)
        })
    }

    #[inline]
    pub fn contains_key(&self, key: &str) -> bool {
        self.position(key).is_some()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Entries in key order.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> + Clone {
        self.0.iter().map(|(k, v)| (&**k, v))
    }

    /// Keys in sorted order.
    #[inline]
    pub fn keys(&self) -> impl Iterator<Item = &str> + Clone {
        self.0.iter().map(|(k, _)| &**k)
    }

    /// Values, in their keys' sorted order.
    #[inline]
    pub fn values(&self) -> impl Iterator<Item = &Value> + Clone {
        self.0.iter().map(|(_, v)| v)
    }

    /// Insert or replace, returning the previous value.
    ///
    /// This is the rare path: it takes ownership of the backing vector
    /// (`Arc::make_mut`, so it deep-copies only when the map is actually
    /// shared) and shifts to keep the sort. Bulk construction goes through
    /// [`PropMap::from_pairs`] instead — repeated `insert` on a large map is
    /// O(n²).
    pub fn insert(&mut self, key: impl Into<PropKey>, value: Value) -> Option<Value> {
        let key = key.into();
        let entries = Arc::make_mut(&mut self.0);
        match entries.binary_search_by(|(k, _)| (**k).cmp(&key)) {
            Ok(i) => Some(std::mem::replace(&mut entries[i].1, value)),
            Err(i) => {
                entries.insert(i, (key, value));
                None
            }
        }
    }

    /// Remove a key, returning its value.
    pub fn remove(&mut self, key: &str) -> Option<Value> {
        let entries = Arc::make_mut(&mut self.0);
        match entries.binary_search_by(|(k, _)| (**k).cmp(key)) {
            Ok(i) => Some(entries.remove(i).1),
            Err(_) => None,
        }
    }

    /// Mutable access to a value already present.
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        let i = self.position(key)?;
        Some(&mut Arc::make_mut(&mut self.0)[i].1)
    }

    /// Keep only the entries the predicate accepts.
    pub fn retain(&mut self, mut f: impl FnMut(&str, &Value) -> bool) {
        Arc::make_mut(&mut self.0).retain(|(k, v)| f(k, v));
    }

    /// Consume into the owned pair vector, avoiding the copy when this handle
    /// is the only owner.
    pub fn into_pairs(self) -> Vec<(PropKey, Value)> {
        Arc::try_unwrap(self.0).unwrap_or_else(|arc| (*arc).clone())
    }

    #[inline]
    fn position(&self, key: &str) -> Option<usize> {
        self.0.binary_search_by(|(k, _)| (**k).cmp(key)).ok()
    }
}

/// Keep the **last** entry of each equal-key run. `Vec::dedup_by` keeps the
/// first, which would make a re-inserted key lose to the value it replaced.
fn dedup_keep_last(pairs: &mut Vec<(PropKey, Value)>) {
    if pairs.len() < 2 {
        return;
    }
    let mut write = 0usize;
    for read in 1..pairs.len() {
        if pairs[read].0 == pairs[write].0 {
            pairs.swap(write, read);
        } else {
            write += 1;
            pairs.swap(write, read);
        }
    }
    pairs.truncate(write + 1);
}

/// Borrowed iteration yields `(&str, &Value)`, so `for (k, v) in &map` reads
/// the same as it did over a `BTreeMap` without the `&String` indirection.
pub type PropMapIter<'a> = std::iter::Map<
    std::slice::Iter<'a, (PropKey, Value)>,
    fn(&'a (PropKey, Value)) -> (&'a str, &'a Value),
>;

impl<'a> IntoIterator for &'a PropMap {
    type Item = (&'a str, &'a Value);
    type IntoIter = PropMapIter<'a>;
    fn into_iter(self) -> Self::IntoIter {
        fn split(e: &(PropKey, Value)) -> (&str, &Value) {
            (&e.0, &e.1)
        }
        self.0
            .iter()
            .map(split as fn(&'a (PropKey, Value)) -> (&'a str, &'a Value))
    }
}

impl IntoIterator for PropMap {
    type Item = (PropKey, Value);
    type IntoIter = std::vec::IntoIter<(PropKey, Value)>;
    fn into_iter(self) -> Self::IntoIter {
        self.into_pairs().into_iter()
    }
}

impl FromIterator<(PropKey, Value)> for PropMap {
    fn from_iter<T: IntoIterator<Item = (PropKey, Value)>>(iter: T) -> Self {
        Self::from_pairs(iter.into_iter().collect())
    }
}

impl<'a> FromIterator<(&'a str, Value)> for PropMap {
    fn from_iter<T: IntoIterator<Item = (&'a str, Value)>>(iter: T) -> Self {
        Self::from_pairs(
            iter.into_iter()
                .map(|(k, v)| (PropKey::from(k), v))
                .collect(),
        )
    }
}

impl From<BTreeMap<String, Value>> for PropMap {
    /// A `BTreeMap` is already sorted and de-duplicated, so this is a straight
    /// re-key with no sort.
    fn from(map: BTreeMap<String, Value>) -> Self {
        Self::from_sorted_pairs(
            map.into_iter()
                .map(|(k, v)| (PropKey::from(k), v))
                .collect(),
        )
    }
}

impl From<PropMap> for BTreeMap<String, Value> {
    fn from(map: PropMap) -> Self {
        map.into_pairs()
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect()
    }
}

// ============================================================================
// Serde — postcard map framing, byte-identical to `BTreeMap<String, Value>`
// ============================================================================

impl Serialize for PropMap {
    /// Emits a serde **map**, in key order, exactly as `BTreeMap` does. Under
    /// postcard that is a varint length followed by the entries; under JSON it
    /// is an object. Pinned by `value_byte_identity_tests`.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (k, v) in self.0.iter() {
            map.serialize_entry(&**k, v)?;
        }
        map.end()
    }
}

/// Deserializes a key straight into an `Arc<str>`.
///
/// `String` then `Arc::from` would allocate twice per key; `visit_str` on the
/// borrowed bytes postcard hands back allocates once.
struct PropKeySeed;

impl<'de> Deserialize<'de> for PropKeyWrapper {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_str(PropKeySeed)
    }
}

struct PropKeyWrapper(PropKey);

impl<'de> Visitor<'de> for PropKeySeed {
    type Value = PropKeyWrapper;
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a property key string")
    }
    fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
        Ok(PropKeyWrapper(PropKey::from(v)))
    }
    fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Self::Value, E> {
        Ok(PropKeyWrapper(PropKey::from(v)))
    }
}

impl<'de> Deserialize<'de> for PropMap {
    /// Accepts any serde map. The stored order is already sorted for anything
    /// this engine wrote, but foreign input (JSON parameters, a hand-built
    /// payload) is not, so the result is normalised rather than trusted — the
    /// binary-search invariant is not something a caller can opt out of.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct MapVisitor;
        impl<'de> Visitor<'de> for MapVisitor {
            type Value = PropMap;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a map of property names to values")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut access: A) -> Result<PropMap, A::Error> {
                let mut pairs: Vec<(PropKey, Value)> =
                    Vec::with_capacity(access.size_hint().unwrap_or(0).min(64));
                while let Some((PropKeyWrapper(k), v)) =
                    access.next_entry::<PropKeyWrapper, Value>()?
                {
                    pairs.push((k, v));
                }
                // Sorted input is the overwhelmingly common case (everything
                // this engine writes); check before paying for a sort.
                if pairs.windows(2).all(|w| w[0].0 < w[1].0) {
                    Ok(PropMap::from_sorted_pairs(pairs))
                } else {
                    Ok(PropMap::from_pairs(pairs))
                }
            }
        }
        deserializer.deserialize_map(MapVisitor)
    }
}

#[cfg(test)]
mod tests;
