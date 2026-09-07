//! Freshness state for a disk graph's persistent index bundles.
//!
//! A `property_index_*` / `global_index_*` bundle is an mmap snapshot of the
//! graph at build time, and nothing maintains it afterwards: rewriting the
//! `keys`/`offsets`/`ids` files under a live mapping on every `SET` is exactly
//! the unbounded per-write cost the disk backend exists to avoid. So the
//! bundle's *answer contract* carries the maintenance instead — `Some(v)` may
//! only be returned by a bundle that provably covers every live slot, and a
//! bundle that does not declines with `None`, which the matcher already reads
//! as "scan the type".
//!
//! This is the mapped backend's invalidate-and-rebuild-lazily
//! (`MappedGraph::invalidate_property_index`) with the rebuild moved off the
//! read path: a disk rebuild writes files and needs `&mut`, so a reader can
//! only decline. `reindex()` and `save()` are where the rebuild happens.
//!
//! # Why one [`IndexFreshness`] per bundle, plus a baseline
//!
//! [`IndexFreshness`] reads node creation out of the graph's slot bound rather
//! than out of notifications, so per-bundle state is what makes rebuilding one
//! bundle stop lying about the others. But a bundle is opened **lazily**, on
//! the first lookup that wants it, which can be long after the writes it
//! missed. `baseline` is the freshness a bundle would have had if it had been
//! registered at open time — it is notified like a registered bundle and
//! cloned into every lazily discovered one, so a bundle carried in from a
//! published generation inherits every write since the graph was opened
//! instead of being born fresh.
//!
//! # The tracked gate
//!
//! Every write funnel asks [`DiskIndexFreshness::tracks_anything`] first. It is
//! a `OnceLock` over "does this graph's published generation hold any bundle?"
//! — self-initialising rather than set by each of `DiskGraph`'s constructors,
//! because the answer has to be right *before* the first mutation and a
//! constructor this module does not know about would silently answer `false`.
//! A graph that has never been saved and has no index pays one relaxed load per
//! written row and nothing else.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, OnceLock, RwLock};

use crate::graph::index_freshness::IndexFreshness;

use super::property_index;

/// Freshness for every persistent bundle a `DiskGraph` may consult.
///
/// Keys mirror the two caches on `DiskGraph`: typed bundles by
/// `(node_type, property)`, global bundles by property.
#[derive(Debug)]
pub(crate) struct DiskIndexFreshness {
    /// Freshness a not-yet-registered bundle inherits — see the module docs.
    baseline: Arc<IndexFreshness>,
    typed: RwLock<HashMap<(String, String), Arc<IndexFreshness>>>,
    global: RwLock<HashMap<String, Arc<IndexFreshness>>>,
    /// `Some(true)` once any bundle is known to exist. Latched, never cleared:
    /// a graph that has held a bundle keeps paying the (single-load) gate, and
    /// clearing it would open the window a lazily-opened legacy bundle needs.
    tracked: OnceLock<bool>,
}

impl DiskIndexFreshness {
    /// Freshness for a graph whose bundles all cover slots below `node_bound`.
    pub(crate) fn covering(node_bound: u32) -> Self {
        Self {
            baseline: Arc::new(IndexFreshness::covering(node_bound, None)),
            typed: RwLock::new(HashMap::new()),
            global: RwLock::new(HashMap::new()),
            tracked: OnceLock::new(),
        }
    }

    /// Whether this graph has any persistent bundle to keep honest — the write
    /// path's whole cost when it does not.
    ///
    /// `data_dir` is the published generation the graph was opened on; a
    /// bundle built later latches the answer through [`Self::mark_tracked`].
    #[inline]
    pub(crate) fn tracks_anything(&self, data_dir: &Path) -> bool {
        *self
            .tracked
            .get_or_init(|| directory_holds_a_bundle(data_dir))
    }

    /// Latch "this graph has a bundle" without a directory scan — used by the
    /// builders, which have just written one.
    pub(crate) fn mark_tracked(&self) {
        let _ = self.tracked.set(true);
    }

    /// A node of `node_type` was created at `slot`.
    pub(crate) fn note_created(&self, slot: u32, node_type: &str) {
        // The baseline stands in for bundles of unknown identity, so every
        // creation is a covered one there — the safe direction.
        self.baseline.note_created(slot, true);
        for (key, freshness) in self.typed.read().unwrap_or_else(|e| e.into_inner()).iter() {
            freshness.note_created(slot, key.0 == node_type);
        }
        for freshness in self
            .global
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .values()
        {
            freshness.note_created(slot, true);
        }
    }

    /// A property of the node at `slot` was written. `node_type` is `None` from
    /// a caller that did not resolve it, which marks every typed bundle.
    ///
    /// Field-blind: a typed bundle records `(node_type, property)`, and the
    /// callers that *do* know the written field pass an alias-resolved
    /// spelling that a bundle keyed on the user's spelling cannot be compared
    /// against without re-resolving per row. Marking one extra bundle costs one
    /// declined lookup until the next rebuild; missing one is a wrong answer.
    pub(crate) fn note_property_written(&self, slot: u32, node_type: Option<&str>) {
        self.baseline.note_changed(slot);
        for (key, freshness) in self.typed.read().unwrap_or_else(|e| e.into_inner()).iter() {
            if node_type.is_none_or(|written| written == key.0) {
                freshness.note_changed(slot);
            }
        }
        for freshness in self
            .global
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .values()
        {
            freshness.note_changed(slot);
        }
    }

    /// The node at `slot` was removed.
    ///
    /// Deletion has to mark on its own rather than lean on the tombstone
    /// filter downstream: the filter drops a dead slot, but a slot the graph
    /// hands back out to a node with a *different* indexed value is still
    /// answered from the stale bundle, and one of the matcher's index arms
    /// returns its hits unfiltered.
    pub(crate) fn note_removed(&self, slot: u32) {
        self.note_property_written(slot, None);
    }

    /// Whether the typed bundle for `key` covers the graph up to `node_bound`.
    pub(crate) fn typed_is_fresh(&self, key: &(String, String), node_bound: u32) -> bool {
        let registered = {
            let read = self.typed.read().unwrap_or_else(|e| e.into_inner());
            read.get(key).cloned()
        };
        let freshness = match registered {
            Some(freshness) => freshness,
            None => {
                let inherited = Arc::new((*self.baseline).clone());
                let mut write = self.typed.write().unwrap_or_else(|e| e.into_inner());
                Arc::clone(write.entry(key.clone()).or_insert(inherited))
            }
        };
        !freshness.is_stale(node_bound)
    }

    /// Whether the global bundle for `property` covers the graph up to
    /// `node_bound`.
    pub(crate) fn global_is_fresh(&self, property: &str, node_bound: u32) -> bool {
        let registered = {
            let read = self.global.read().unwrap_or_else(|e| e.into_inner());
            read.get(property).cloned()
        };
        let freshness = match registered {
            Some(freshness) => freshness,
            None => {
                let inherited = Arc::new((*self.baseline).clone());
                let mut write = self.global.write().unwrap_or_else(|e| e.into_inner());
                Arc::clone(write.entry(property.to_string()).or_insert(inherited))
            }
        };
        !freshness.is_stale(node_bound)
    }

    /// Record that the typed bundle for `key` was just built over every slot
    /// below `node_bound`.
    pub(crate) fn mark_typed_built(&self, key: (String, String), node_bound: u32) {
        self.mark_tracked();
        self.typed
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key, Arc::new(IndexFreshness::covering(node_bound, None)));
    }

    /// Global counterpart of [`Self::mark_typed_built`].
    pub(crate) fn mark_global_built(&self, property: &str, node_bound: u32) {
        self.mark_tracked();
        self.global
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(
                property.to_string(),
                Arc::new(IndexFreshness::covering(node_bound, None)),
            );
    }

    /// Drop a typed bundle's state — the bundle itself is gone.
    pub(crate) fn forget_typed(&self, key: &(String, String)) {
        self.typed
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .remove(key);
    }
}

impl Clone for DiskIndexFreshness {
    /// Deep, never shared — a clone is another graph, and `DiskGraph::clone`
    /// empties the bundle caches, so its bundles are all re-discovered from the
    /// baseline this copy carries.
    fn clone(&self) -> Self {
        fn deep<K: Clone + Eq + std::hash::Hash>(
            map: &RwLock<HashMap<K, Arc<IndexFreshness>>>,
        ) -> RwLock<HashMap<K, Arc<IndexFreshness>>> {
            let read = map.read().unwrap_or_else(|e| e.into_inner());
            RwLock::new(
                read.iter()
                    .map(|(key, freshness)| (key.clone(), Arc::new((**freshness).clone())))
                    .collect(),
            )
        }
        let tracked = OnceLock::new();
        if let Some(value) = self.tracked.get() {
            let _ = tracked.set(*value);
        }
        Self {
            baseline: Arc::new((*self.baseline).clone()),
            typed: deep(&self.typed),
            global: deep(&self.global),
            tracked,
        }
    }
}

/// Whether `data_dir` holds any persistent index bundle. One `read_dir` per
/// graph, taken once at the first write.
fn directory_holds_a_bundle(data_dir: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(data_dir) else {
        return false;
    };
    entries.flatten().any(|entry| {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        property_index::is_bundle_file_name(&name)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> (String, String) {
        ("Doc".to_string(), "tag".to_string())
    }

    #[test]
    fn a_lazily_discovered_bundle_inherits_the_writes_it_missed() {
        let state = DiskIndexFreshness::covering(4);
        // A write below the watermark lands only in the baseline — nothing is
        // registered yet.
        state.note_property_written(1, Some("Doc"));

        assert!(
            !state.typed_is_fresh(&key(), 4),
            "a bundle opened after the write must inherit it"
        );
    }

    #[test]
    fn a_rebuild_clears_only_its_own_bundle() {
        let state = DiskIndexFreshness::covering(4);
        assert!(state.typed_is_fresh(&key(), 4));
        assert!(state.global_is_fresh("title", 4));

        state.note_created(4, "Doc");
        assert!(!state.typed_is_fresh(&key(), 5));
        assert!(!state.global_is_fresh("title", 5));

        state.mark_typed_built(key(), 5);
        assert!(state.typed_is_fresh(&key(), 5));
        assert!(
            !state.global_is_fresh("title", 5),
            "rebuilding one bundle says nothing about another"
        );
    }

    #[test]
    fn a_creation_of_an_uncovered_type_leaves_the_typed_bundle_fresh() {
        let state = DiskIndexFreshness::covering(4);
        assert!(state.typed_is_fresh(&key(), 4), "register the bundle");

        state.note_created(4, "Person");

        assert!(
            state.typed_is_fresh(&key(), 5),
            "bulk-loading another type must not stale this one"
        );
        assert!(
            !state.global_is_fresh("title", 5),
            "a cross-type bundle covers it"
        );
    }

    #[test]
    fn a_removal_stales_every_bundle() {
        let state = DiskIndexFreshness::covering(4);
        assert!(state.typed_is_fresh(&key(), 4));

        state.note_removed(2);

        assert!(!state.typed_is_fresh(&key(), 4));
        assert!(!state.global_is_fresh("title", 4));
    }

    #[test]
    fn a_clone_shares_no_state_with_its_source() {
        let state = DiskIndexFreshness::covering(4);
        assert!(state.typed_is_fresh(&key(), 4));
        let copy = state.clone();

        copy.note_property_written(1, None);

        assert!(!copy.typed_is_fresh(&key(), 4));
        assert!(state.typed_is_fresh(&key(), 4), "the source must not move");
    }
}
