//! [`RecordRef`] — a node or edge record handed out by a storage read.
//!
//! The heap backends hold their `NodeData` / `EdgeData` and lend them out; the
//! disk backend has no resident record and builds one per read. A single
//! return type covers both: the lifetime of the record travels with the value,
//! so nothing a caller holds can be freed behind its back.
//!
//! The handle keeps a direct pointer beside its optional owner rather than
//! being an enum, so reading through it is a single load with no branch.
//! Scans and sorts deref it once per property read; an enum handle measured
//! 8–9 % slower on an in-memory `ORDER BY` over a timestamp column.

use std::fmt;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ptr::NonNull;

/// A record that is either borrowed from the graph or materialized for the
/// caller. Dereferences to the record in both cases.
pub struct RecordRef<'a, T> {
    /// Into the graph (lent) or at `owner`'s allocation.
    ptr: NonNull<T>,
    /// The allocation `ptr` points at when the record was built for the read.
    /// A separate type with no `'a`, so dropping a handle frees it without
    /// counting as a use of the graph borrow (`RecordRef` has no `Drop` of its
    /// own, exactly like the `&'a T` it stands in for).
    owner: Option<Owned<T>>,
    _lent: PhantomData<&'a T>,
}

/// An allocation leaked from a `Box` and freed on drop.
struct Owned<T>(NonNull<T>, PhantomData<Box<T>>);

impl<T> Drop for Owned<T> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: the pointer came from `Box::leak` in `RecordRef::owned` and
        // is freed exactly once, here.
        drop(unsafe { Box::from_raw(self.0.as_ptr()) });
    }
}

// SAFETY: a lent handle is a `&'a T` (Send when `T: Sync`); an owned one is a
// `Box<T>` (Send when `T: Send`). Requiring both covers either state. Shared
// access only ever hands out `&T`, so `Sync` needs `T: Sync`.
unsafe impl<T: Send + Sync> Send for RecordRef<'_, T> {}
unsafe impl<T: Sync> Sync for RecordRef<'_, T> {}

impl<'a, T> RecordRef<'a, T> {
    /// A record lent by the graph, valid for the graph borrow.
    #[inline]
    pub fn borrowed(record: &'a T) -> Self {
        RecordRef {
            ptr: NonNull::from(record),
            owner: None,
            _lent: PhantomData,
        }
    }

    /// A record built for this read; freed when the handle drops.
    #[inline]
    pub fn owned(record: Box<T>) -> Self {
        let ptr = NonNull::from(Box::leak(record));
        RecordRef {
            ptr,
            owner: Some(Owned(ptr, PhantomData)),
            _lent: PhantomData,
        }
    }

    /// Whether the record was built for this read (disk) rather than lent.
    #[inline]
    pub fn is_owned(&self) -> bool {
        self.owner.is_some()
    }

    /// The graph's own reference, for a lent record; `None` for one built for
    /// the read, which has no `'a` reference to give.
    #[inline]
    pub fn as_lent(&self) -> Option<&'a T> {
        if self.owner.is_some() {
            None
        } else {
            // SAFETY: a lent `ptr` came from a `&'a T` in `borrowed`.
            Some(unsafe { self.ptr.as_ref() })
        }
    }

    /// The lent reference, for callers over a heap backend, whose records are
    /// always lent. A handle built for the read (disk) has no `'a` reference
    /// to give, so reaching it here is a backend mix-up.
    #[inline]
    pub(crate) fn lent(self) -> &'a T {
        self.as_lent().expect("a heap backend lends its records")
    }
}

impl<T> Deref for RecordRef<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: lent, `ptr` is a `&'a T` that outlives `self`; owned, it is
        // the live allocation `self.owner` frees only when `self` drops.
        unsafe { self.ptr.as_ref() }
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
        RecordRef::borrowed(r)
    }
}

// Run under Miri in CI (`miri-loaders`): the handle is the one place the read
// path holds a raw pointer.
#[cfg(test)]
mod tests {
    use super::RecordRef;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static DROPS: AtomicUsize = AtomicUsize::new(0);

    struct Counted(String);
    impl Drop for Counted {
        fn drop(&mut self) {
            DROPS.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn record_ref_lends_or_owns_and_frees_only_what_it_owns() {
        let lent = Counted("lent".into());
        let before = DROPS.load(Ordering::SeqCst);
        {
            let r = RecordRef::borrowed(&lent);
            assert!(!r.is_owned());
            assert_eq!(r.0, "lent");
            assert!(std::ptr::eq(r.as_lent().unwrap(), &lent));
            // Moving the handle keeps the borrow valid.
            let moved = r;
            assert_eq!(moved.0, "lent");
        }
        assert_eq!(
            DROPS.load(Ordering::SeqCst),
            before,
            "a lent record is not dropped"
        );

        {
            let r = RecordRef::owned(Box::new(Counted("owned".into())));
            assert!(r.is_owned() && r.as_lent().is_none());
            let moved = r;
            let inner: &Counted = &moved;
            assert_eq!(inner.0, "owned");
        }
        assert_eq!(
            DROPS.load(Ordering::SeqCst),
            before + 1,
            "an owned record is dropped once"
        );
        drop(lent);
    }

    #[test]
    fn record_refs_cross_threads() {
        let handles: Vec<RecordRef<'static, String>> = (0..4)
            .map(|i| RecordRef::owned(Box::new(format!("r{i}"))))
            .collect();
        let joined = std::thread::spawn(move || handles.iter().map(|h| h.len()).sum::<usize>())
            .join()
            .unwrap();
        assert_eq!(joined, 8);
    }
}
