//! Counted copy-on-write for the schema-scale state a rollback shell shares.
//!
//! One primitive, used by two owners: the six `Arc` maps on `DirGraph`
//! (`dir_graph::schema_cow`, which also documents the contract and the
//! measurement) and the `StringInterner`'s key→name table
//! (`storage::interner`). Both are cloned by `rollback::schema_shell` before
//! every mutating statement and restored verbatim on failure, which is exactly
//! the shape `Arc` + copy-on-write is for: a pointer to the pre-statement
//! value is as good as a copy of it.
//!
//! The counter is why the primitive is shared rather than open-coded at each
//! site. It is the fourth member of the oracle family — `BACKEND_CLONE_NODES`,
//! `JOURNAL_NODE_PRE_IMAGES`, `COLUMN_STORE_CLONES` are the others — and it
//! sees what none of them can: the schema catalogue lives in neither a
//! backend, a `NodeData`, nor a `ColumnStore`, so a per-statement copy of it
//! read zero on all three for as long as it existed.

use std::sync::Arc;

#[cfg(test)]
thread_local! {
    /// Schema-state deep copies (`Arc::make_mut` on a shared handle) since the
    /// last reset. Thread-local like its sibling oracles: statement-scoped
    /// writes happen on the calling thread.
    static SCHEMA_MAP_FORKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_schema_map_forks() {
    SCHEMA_MAP_FORKS.set(0);
}

/// Schema-state deep copies on this thread since the last reset.
#[cfg(test)]
pub(crate) fn schema_map_forks() -> usize {
    SCHEMA_MAP_FORKS.get()
}

/// `Arc::make_mut`, counted.
///
/// The count condition mirrors `make_mut`'s own: it copies when another strong
/// handle exists, and also when a `Weak` does (to leave the weak dangling).
#[inline]
pub(crate) fn cow_mut<T: Clone>(slot: &mut Arc<T>) -> &mut T {
    #[cfg(test)]
    if Arc::strong_count(slot) > 1 || Arc::weak_count(slot) > 0 {
        SCHEMA_MAP_FORKS.set(SCHEMA_MAP_FORKS.get() + 1);
    }
    Arc::make_mut(slot)
}
