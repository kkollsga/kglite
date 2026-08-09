//! Test-only drop-order seam for temp directories that back a live
//! disk/mapped graph.
//!
//! **Why a seam at all.** On Unix a mapped file can be unlinked while its
//! mapping is still live: the inode survives until the last mapping goes
//! away. So a test that removes a temp directory *before* dropping the graph
//! mapped into it still passes — the path is gone, nothing crashes, and the
//! ordering bug is invisible. "Removed the directory and nothing blew up" is
//! therefore a vacuous assertion: the mutation that restores the wrong order
//! stays green.
//!
//! The disk backend's `debug_assert_arena_guard_active` protocol does not
//! cover this. It tracks *query* lifetime against arena reclamation inside a
//! live `DiskGraph`; it says nothing about whether the graph itself still
//! exists when its backing directory is deleted.
//!
//! **What this provides instead.** A graph that maps into the directory is
//! wrapped in a [`TrackedOwner`], which carries an `Arc<()>` liveness token;
//! [`TempGraphDir::watch`] keeps the matching `Weak<()>`. At the moment the
//! directory is removed — explicitly via [`TempGraphDir::remove_now`], or
//! implicitly when the guard drops — every watched token must be dead. A
//! live token means an owner outlived the directory (typically because it
//! was *declared* before it, so reverse-declaration drop order destroys the
//! directory first), and the test fails naming that owner.
//!
//! Because the owner and the guard are constructed independently, swapping
//! their two `let` bindings is enough to make the seam red: the ordering
//! itself is the asserted property, on every platform, not only on Windows
//! where the OS happens to refuse the removal.

use std::cell::RefCell;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::sync::Arc;

use tempfile::TempDir;

/// A value whose drop a [`TempGraphDir`] can observe.
///
/// Transparent: `Deref`/`DerefMut` forward to the wrapped value, so tests
/// keep calling methods and touching fields as before. Field order matters —
/// `value` is declared first, so it drops before the liveness token, making
/// the token a faithful "this owner is still alive" signal.
pub(crate) struct TrackedOwner<T> {
    value: T,
    liveness: Arc<()>,
    label: &'static str,
}

impl<T> TrackedOwner<T> {
    /// Wrap `value`; `label` names it in the drop-order failure message.
    pub(crate) fn new(label: &'static str, value: T) -> Self {
        Self {
            value,
            liveness: Arc::new(()),
            label,
        }
    }

    /// Consume the wrapper and return the inner value, giving up tracking.
    #[allow(dead_code)]
    pub(crate) fn into_inner(self) -> T {
        self.value
    }
}

impl<T> Deref for TrackedOwner<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T> DerefMut for TrackedOwner<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

/// `(label, is-the-owner-still-alive?)`. Boxed so a probe can be a `Weak<()>`
/// from a [`TrackedOwner`] or a `Weak<T>` from a caller-held `Arc<T>`.
type OwnerProbe = (&'static str, Box<dyn Fn() -> bool>);

/// A temp directory plus the drop-order contract for everything mapped into
/// it: every watched owner must be dropped before the directory is removed.
pub(crate) struct TempGraphDir {
    dir: Option<TempDir>,
    watched: RefCell<Vec<OwnerProbe>>,
}

impl TempGraphDir {
    pub(crate) fn new() -> Self {
        Self {
            dir: Some(TempDir::new().expect("failed to create temp dir")),
            watched: RefCell::new(Vec::new()),
        }
    }

    /// Path of the temp directory. Panics after [`Self::remove_now`].
    pub(crate) fn path(&self) -> &Path {
        self.dir
            .as_ref()
            .expect("temp dir already removed by remove_now()")
            .path()
    }

    /// Require `owner` to be dropped before this directory is removed.
    pub(crate) fn watch<T>(&self, owner: &TrackedOwner<T>) {
        let token = Arc::downgrade(&owner.liveness);
        self.watched
            .borrow_mut()
            .push((owner.label, Box::new(move || token.strong_count() > 0)));
    }

    /// Wrap `value` and watch it in one step. Because the guard must already
    /// exist to call this, the owner is necessarily declared *after* the
    /// directory — i.e. it drops first, which is the ordering being asserted.
    pub(crate) fn own<T>(&self, label: &'static str, value: T) -> TrackedOwner<T> {
        let owner = TrackedOwner::new(label, value);
        self.watch(&owner);
        owner
    }

    /// Require every clone of `shared` to be gone before this directory is
    /// removed. Stronger than [`Self::watch`] for `Arc`-held owners: the
    /// probe is the `Arc`'s own strong count, so a clone leaked into a
    /// thread or a stray binding is caught too, not just the handle the
    /// fixture holds.
    pub(crate) fn watch_arc<T: 'static>(&self, label: &'static str, shared: &Arc<T>) {
        let token = Arc::downgrade(shared);
        self.watched
            .borrow_mut()
            .push((label, Box::new(move || token.strong_count() > 0)));
    }

    /// Remove the directory now, after asserting every watched owner has
    /// already been dropped. Use where a test previously called
    /// `remove_dir_all` explicitly.
    ///
    /// Removal errors are surfaced, not swallowed: Windows refuses to delete
    /// a file with a live mapped view (`ERROR_USER_MAPPED_FILE`), so an error
    /// here is supplemental platform evidence for the same ordering bug the
    /// liveness tokens catch everywhere.
    pub(crate) fn remove_now(mut self) {
        self.assert_owners_dropped("removing the temp directory");
        if let Some(dir) = self.dir.take() {
            dir.close().expect("temp dir removal failed");
        }
    }

    fn live_owners(&self) -> Vec<&'static str> {
        self.watched
            .borrow()
            .iter()
            .filter(|(_, still_alive)| still_alive())
            .map(|(label, _)| *label)
            .collect()
    }

    fn assert_owners_dropped(&self, action: &str) {
        let live = self.live_owners();
        assert!(
            live.is_empty(),
            "drop-order violation: {action} while {} owner(s) still hold mappings into it: \
             {live:?}. Drop the graph/mapped owner first — declare the temp dir before it, or \
             call drop() explicitly before cleanup.",
            live.len(),
        );
    }
}

impl Drop for TempGraphDir {
    fn drop(&mut self) {
        // A failed assertion elsewhere already unwound the test; asserting
        // again here would abort the process instead of reporting it.
        if std::thread::panicking() {
            return;
        }
        self.assert_owners_dropped("dropping the temp directory");
    }
}

#[cfg(test)]
mod seam_tests {
    use super::*;

    /// The seam itself must be non-vacuous: a directory removed while its
    /// owner is alive has to fail. Verified out-of-band (the assertion is in
    /// `Drop`, so it is caught rather than observed as a test failure).
    #[test]
    fn removal_while_owner_alive_is_detected() {
        let result = std::panic::catch_unwind(|| {
            let owner = TrackedOwner::new("probe owner", ());
            let dir = TempGraphDir::new();
            dir.watch(&owner);
            dir.remove_now();
            drop(owner);
        });
        let payload = result.expect_err("removal with a live owner must panic");
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .unwrap_or("<non-string panic>");
        assert!(
            message.contains("drop-order violation"),
            "unexpected panic: {message}"
        );
    }

    /// `watch_arc` must catch a clone the fixture does not hold — the case
    /// that makes an `Arc`-shared graph's safety non-accidental.
    #[test]
    fn surviving_arc_clone_is_detected() {
        let result = std::panic::catch_unwind(|| {
            let dir = TempGraphDir::new();
            let shared = Arc::new(7u8);
            dir.watch_arc("shared probe", &shared);
            let leaked = Arc::clone(&shared);
            drop(shared);
            dir.remove_now();
            drop(leaked);
        });
        let payload = result.expect_err("a surviving Arc clone must panic");
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .unwrap_or("<non-string panic>");
        assert!(
            message.contains("drop-order violation"),
            "unexpected panic: {message}"
        );
    }

    #[test]
    fn owner_dropped_first_is_accepted() {
        let dir = TempGraphDir::new();
        let owner = TrackedOwner::new("probe owner", ());
        dir.watch(&owner);
        drop(owner);
        dir.remove_now();
    }
}
