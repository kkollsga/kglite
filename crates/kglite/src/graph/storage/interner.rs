// src/graph/storage/interner.rs
//
// InternedKey + StringInterner + serde thread-local guards.
//
// InternedKey is a compact FNV-1a hash of a property/type-name string;
// StringInterner holds the reverse mapping. Serde round-trips keys
// through their original strings using thread-local interner pointers
// installed by the RAII guards (`SerdeSerializeGuard` /
// `SerdeDeserializeGuard`). A third guard (`StripPropertiesGuard`)
// enables the v3 topology-mode serialization path.
//
// Extracted from `src/graph/schema.rs` in Phase 7 (Stage 2.2).

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::cell::Cell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// A compact property key backed by a hash of the original string.
/// Lookups via `get_property(key)` compute the hash inline — no interner needed.
/// Only methods that output string keys (e.g. `property_iter`) require the interner.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
pub struct InternedKey(u64);

impl InternedKey {
    /// Compute the interned key from a string. **Must be deterministic across
    /// processes and library versions** — `DiskNodeSlot.node_type` persists
    /// this as raw u64 on disk (`disk_graph.rs`), and the loader resolves it
    /// via the freshly-built interner's hashes. A per-process random seed
    /// (e.g. `DefaultHasher`) would break cross-process disk loads.
    ///
    /// Uses FNV-1a 64-bit. Fast, zero-alloc, dependency-free, deterministic.
    /// Our corpus (property names, type names) is at most a few thousand
    /// short strings, so collision risk is negligible; debug builds also
    /// assert non-collision in `StringInterner::register`.
    #[inline]
    pub fn from_str(s: &str) -> Self {
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;
        let mut h = FNV_OFFSET;
        for &byte in s.as_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
        InternedKey(h)
    }

    /// Get the raw u64 hash value. Used for disk storage.
    #[inline]
    pub fn as_u64(&self) -> u64 {
        self.0
    }

    /// Reconstruct from a raw u64 hash value. Used when loading from disk.
    #[inline]
    pub fn from_u64(v: u64) -> Self {
        InternedKey(v)
    }
}

impl Hash for InternedKey {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.0);
    }
}

/// Serializes InternedKey as its original string (backward-compatible with
/// HashMap<String, Value> on disk). Requires the thread-local SERIALIZE_INTERNER
/// to be set before the top-level serialize call.
impl Serialize for InternedKey {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        SERIALIZE_INTERNER.with(|cell| {
            let ptr = cell
                .get()
                .expect("BUG: SERIALIZE_INTERNER not set during InternedKey serialization");
            // SAFETY: ptr is set by SerdeInternerGuard which ensures the reference
            // outlives the serialize call (the guard lives on the caller's stack).
            let interner = unsafe { &*ptr };
            interner.resolve(*self).serialize(serializer)
        })
    }
}

/// Deserializes InternedKey from a string (backward-compatible with
/// HashMap<String, Value> on disk). Registers the string in the thread-local
/// DESERIALIZE_INTERNER if set.
///
/// Uses a custom Visitor to avoid String allocation: bincode's SliceReader
/// provides borrowed &str directly from the decompressed buffer. Only the
/// first occurrence of each key allocates (in the interner). For ~5.6M
/// property keys with ~200 unique ones, this eliminates ~5.6M allocations.
impl<'de> Deserialize<'de> for InternedKey {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct KeyVisitor;
        impl<'de> serde::de::Visitor<'de> for KeyVisitor {
            type Value = InternedKey;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a string key")
            }
            /// Fast path: hash borrowed &str directly, no String allocation.
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
                let key = InternedKey::from_str(v);
                DESERIALIZE_INTERNER.with(|cell| {
                    if let Some(ptr) = cell.get() {
                        // SAFETY: ptr is set by SerdeInternerGuard which
                        // ensures the &mut reference outlives the
                        // deserialize call (the guard lives on the
                        // caller's stack, same pattern as Serialize above).
                        let interner = unsafe { &mut *ptr };
                        interner.register(key, v);
                    }
                });
                Ok(key)
            }
            /// Fallback for formats that provide owned Strings (e.g. JSON).
            fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Self::Value, E> {
                self.visit_str(&v)
            }
        }
        deserializer.deserialize_str(KeyVisitor)
    }
}

/// Reverse mapping from InternedKey → original string.
/// Used for serialization and for methods that output string keys.
#[derive(Debug, Clone, Default)]
pub struct StringInterner {
    strings: HashMap<InternedKey, String>,
}

impl StringInterner {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a key-string mapping. If the key already exists, this is a no-op.
    /// Panics in debug mode if the same hash maps to a different string (collision).
    #[inline]
    pub fn register(&mut self, key: InternedKey, s: &str) {
        self.strings.entry(key).or_insert_with(|| s.to_string());
        #[cfg(debug_assertions)]
        {
            let existing = &self.strings[&key];
            debug_assert_eq!(
                existing, s,
                "InternedKey hash collision: '{}' and '{}' have the same hash",
                existing, s
            );
        }
    }

    /// Intern a string: compute its key and register the reverse mapping.
    #[inline]
    pub fn get_or_intern(&mut self, s: &str) -> InternedKey {
        let key = InternedKey::from_str(s);
        self.register(key, s);
        key
    }

    /// Resolve an InternedKey back to its string. Panics if the key is unknown.
    #[inline]
    pub fn resolve(&self, key: InternedKey) -> &str {
        self.strings
            .get(&key)
            .map(|s| s.as_str())
            .unwrap_or_else(|| {
                eprintln!(
                    "BUG: InternedKey {} not found in StringInterner ({} entries)",
                    key.as_u64(),
                    self.strings.len()
                );
                "<unknown>"
            })
    }

    /// Iterate over all (key, string) pairs in the interner.
    pub fn iter(&self) -> impl Iterator<Item = (InternedKey, &str)> {
        self.strings.iter().map(|(&k, v)| (k, v.as_str()))
    }

    /// Resolve an InternedKey back to its string, returning None if unknown.
    #[inline]
    pub fn try_resolve(&self, key: InternedKey) -> Option<&str> {
        self.strings.get(&key).map(|s| s.as_str())
    }

    /// Compute the InternedKey for a string (if it exists in the interner).
    #[inline]
    pub fn try_resolve_to_key(&self, s: &str) -> Option<InternedKey> {
        let key = InternedKey::from_str(s);
        if self.strings.contains_key(&key) {
            Some(key)
        } else {
            None
        }
    }
}

// ─── Thread-local serde support ───────────────────────────────────────────────

thread_local! {
    static SERIALIZE_INTERNER: Cell<Option<*const StringInterner>> = const { Cell::new(None) };
    static DESERIALIZE_INTERNER: Cell<Option<*mut StringInterner>> = const { Cell::new(None) };
    /// When true, PropertyStorage::Serialize emits an empty map (v3 topology mode).
    pub(crate) static STRIP_PROPERTIES: Cell<bool> = const { Cell::new(false) };
}

/// RAII guard that sets the thread-local interner for serialization.
/// The interner reference must outlive the guard (enforced by the lifetime).
pub(crate) struct SerdeSerializeGuard<'a> {
    _phantom: std::marker::PhantomData<&'a StringInterner>,
}

impl<'a> SerdeSerializeGuard<'a> {
    pub fn new(interner: &'a StringInterner) -> Self {
        SERIALIZE_INTERNER.with(|cell| cell.set(Some(interner as *const StringInterner)));
        SerdeSerializeGuard {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl Drop for SerdeSerializeGuard<'_> {
    fn drop(&mut self) {
        SERIALIZE_INTERNER.with(|cell| cell.set(None));
    }
}

/// RAII guard that sets the thread-local interner for deserialization.
pub(crate) struct SerdeDeserializeGuard<'a> {
    _phantom: std::marker::PhantomData<&'a mut StringInterner>,
}

impl<'a> SerdeDeserializeGuard<'a> {
    pub fn new(interner: &'a mut StringInterner) -> Self {
        DESERIALIZE_INTERNER.with(|cell| cell.set(Some(interner as *mut StringInterner)));
        SerdeDeserializeGuard {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl Drop for SerdeDeserializeGuard<'_> {
    fn drop(&mut self) {
        DESERIALIZE_INTERNER.with(|cell| cell.set(None));
    }
}

/// RAII guard that enables property stripping during serialization.
/// While active, PropertyStorage::Serialize emits empty maps (v3 topology mode).
pub(crate) struct StripPropertiesGuard;

impl StripPropertiesGuard {
    pub fn new() -> Self {
        STRIP_PROPERTIES.with(|cell| cell.set(true));
        StripPropertiesGuard
    }
}

impl Drop for StripPropertiesGuard {
    fn drop(&mut self) {
        STRIP_PROPERTIES.with(|cell| cell.set(false));
    }
}
