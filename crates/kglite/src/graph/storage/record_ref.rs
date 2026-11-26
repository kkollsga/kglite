//! [`RecordRef`] — a node or edge record handed out by a storage read.
//!
//! The heap backends hold their `NodeData` / `EdgeData` and lend them out; the
//! disk backend has no resident record and builds one per read. A single
//! return type covers both: the lifetime of the record travels with the value,
//! so nothing a caller holds can be freed behind its back.

use std::fmt;
use std::ops::Deref;

/// A record that is either borrowed from the graph or materialized for the
/// caller. Dereferences to the record in both cases.
pub enum RecordRef<'a, T> {
    /// Lent out by a heap backend; valid for the graph borrow.
    Borrowed(&'a T),
    /// Built for this read (disk backend); freed when the handle drops.
    Owned(Box<T>),
}

impl<'a, T> RecordRef<'a, T> {
    /// The lent reference, for callers over a heap backend, whose records are
    /// always lent. A handle built for the read (disk) has no `'a` reference
    /// to give, so reaching it here is a backend mix-up.
    #[inline]
    pub(crate) fn lent(self) -> &'a T {
        match self {
            RecordRef::Borrowed(r) => r,
            RecordRef::Owned(_) => unreachable!("a heap backend lends its records"),
        }
    }
}

impl<T> Deref for RecordRef<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        match self {
            RecordRef::Borrowed(r) => r,
            RecordRef::Owned(b) => b,
        }
    }
}

impl<T> AsRef<T> for RecordRef<'_, T> {
    #[inline]
    fn as_ref(&self) -> &T {
        self
    }
}

impl<T: fmt::Debug> fmt::Debug for RecordRef<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

impl<'a, T> From<&'a T> for RecordRef<'a, T> {
    #[inline]
    fn from(r: &'a T) -> Self {
        RecordRef::Borrowed(r)
    }
}
