//! Test-only column-store poisoning — the runtime half of the D1 Phase-2 gate.
//!
//! # What it simulates
//!
//! D1 Phase 3 makes the storage backend the sole owner of a type's
//! [`ColumnStore`](crate::graph::storage::column_store::ColumnStore), and every
//! read resolves through the backend rather than through the node's own `Arc`.
//! Until that lands, both routes point at the same object, so a caller that
//! still reads `NodeData` directly is *indistinguishable* from one that reads
//! through [`NodeView`](crate::graph::storage::NodeView) — and a gate that
//! cannot tell them apart is not a gate.
//!
//! Poisoning removes that ambiguity by making the two routes disagree, the same
//! way Phase 3 will:
//!
//! 1. every node of the type is re-pointed at a **stale** private clone;
//! 2. the truth is written into the type's **master** store;
//! 3. the post-write master is installed here as the **authoritative** route.
//!
//! [`NodeView::from_node_data`](crate::graph::storage::NodeView) consults this
//! module first, so a migrated caller sees the truth and an unmigrated one sees
//! the stale replica. `DirGraph::poison_column_store` in
//! `column_ownership_tests` drives all three steps.
//!
//! # Cost when not testing
//!
//! The whole module is `#[cfg(test)]`. `NodeView::from_node_data`'s call to
//! [`authoritative`] compiles to nothing in any non-test build, so the release
//! read path is byte-identical to Phase 1's.
//!
//! # Snapshot semantics
//!
//! [`install`] leaks a snapshot of the master taken at poison time. Writes made
//! *after* the poison are not visible through the authoritative route, so a
//! poisoned test asserts on reads, not on subsequent mutations. That is
//! deliberate: it keeps the hook a few lines rather than a second ownership
//! implementation.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};

use rustc_hash::FxHashMap;

use crate::graph::schema::InternedKey;
use crate::graph::storage::column_store::ColumnStore;

/// Fast reject so the common (unpoisoned) test path pays one relaxed load
/// rather than a thread-local lookup per node view.
static ANY_ACTIVE: AtomicBool = AtomicBool::new(false);

thread_local! {
    static OVERRIDES: RefCell<FxHashMap<InternedKey, &'static ColumnStore>> =
        RefCell::new(FxHashMap::default());
}

/// Restores normal (node-handle) resolution when dropped.
///
/// Hold it for the lifetime of the assertion — `let _guard = …;`, never
/// `let _ = …;`, which drops immediately and silently un-poisons.
#[must_use = "dropping the guard immediately un-poisons the store and makes the test vacuous"]
pub(crate) struct PoisonGuard {
    keys: Vec<InternedKey>,
}

impl Drop for PoisonGuard {
    fn drop(&mut self) {
        OVERRIDES.with(|cell| {
            let mut map = cell.borrow_mut();
            for key in &self.keys {
                map.remove(key);
            }
        });
    }
}

/// Install `store` as the authoritative read route for `type_key`.
///
/// `store` must outlive every `NodeView` built while the guard is alive; callers
/// pass a `Box::leak`ed snapshot, which is bounded by the number of poison calls
/// in a test run.
pub(crate) fn install(type_key: InternedKey, store: &'static ColumnStore) -> PoisonGuard {
    OVERRIDES.with(|cell| {
        cell.borrow_mut().insert(type_key, store);
    });
    ANY_ACTIVE.store(true, Ordering::Relaxed);
    PoisonGuard {
        keys: vec![type_key],
    }
}

/// The authoritative store for `type_key`, when poisoning is active.
#[inline]
pub(crate) fn authoritative(type_key: InternedKey) -> Option<&'static ColumnStore> {
    if !ANY_ACTIVE.load(Ordering::Relaxed) {
        return None;
    }
    OVERRIDES.with(|cell| cell.borrow().get(&type_key).copied())
}
