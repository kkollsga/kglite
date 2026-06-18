//! Loom model of the `Session` commit/snapshot lock pattern.
//!
//! [loom](https://github.com/tokio-rs/loom) exhaustively explores every legal
//! thread interleaving of a small concurrent model — the strongest signal for
//! the exact bug class we fixed in `Session::commit` (a TOCTOU race where the
//! OCC version check and the Arc swap happened under *separate* lock
//! acquisitions, so two committers could both pass the check and both swap).
//!
//! This models the **algorithm**, not the real `Session`: loom requires its
//! own instrumented `loom::sync::{Arc, Mutex}`, and the real `Session` wraps
//! `Arc<DirGraph>` (pervasive across the crate, not feasible to swap under
//! `#[cfg(loom)]`). So the version counter — the thing the race corrupted —
//! stands in for the graph, and the commit logic mirrors
//! `session/transaction.rs::Session::commit` exactly: **one lock guard held
//! across the version read and the swap, with the new version derived from the
//! current value.** If the model is changed to release the lock between the
//! read and the swap (the pre-fix shape), loom finds the interleaving that
//! drops a commit and the assertions below fail.
//!
//! Not part of `cargo test` (needs the loom cfg + dep). Run:
//!   RUSTFLAGS="--cfg loom" cargo test -p kglite --test loom_session
//! See docs/rust/concurrency-verification.md.

#![cfg(loom)]

use loom::sync::{Arc, Mutex};
use loom::thread;

/// Stand-in for the shared graph behind the session: only the monotonic
/// version (`Mutex<Arc<u64>>` mirrors the real `Mutex<Arc<DirGraph>>`).
struct Session {
    inner: Mutex<Arc<u64>>,
}

impl Session {
    fn new() -> Self {
        Session {
            inner: Mutex::new(Arc::new(0)),
        }
    }

    /// Wait-free apart from the momentary lock — mirrors `Session::snapshot`.
    fn version(&self) -> u64 {
        **self.inner.lock().unwrap()
    }

    /// The fixed commit: read the current version and swap the new value in
    /// **under a single held guard** (atomic compare-and-swap). With
    /// `check_occ`, a stale base is rejected so the caller can retry. New
    /// version derives from the current value, so it is monotonic even in
    /// last-writer-wins mode.
    fn commit(&self, base: u64, check_occ: bool) -> Option<u64> {
        let mut guard = self.inner.lock().unwrap();
        let current = **guard;
        if check_occ && current != base {
            return None; // conflict — caller retries
        }
        let new = current + 1;
        *guard = Arc::new(new);
        Some(new)
    }
}

/// Two threads each commit once via OCC + retry (the
/// `concurrent_writers_compose_with_occ_retry` scenario). Every legal
/// interleaving must end at version 2 — no commit lost, none doubled.
#[test]
fn occ_retry_commits_compose() {
    loom::model(|| {
        let s = Arc::new(Session::new());

        let handles: Vec<_> = (0..2)
            .map(|_| {
                let s = s.clone();
                thread::spawn(move || loop {
                    let base = s.version();
                    if s.commit(base, /* check_occ = */ true).is_some() {
                        break;
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(*s.inner.lock().unwrap().as_ref(), 2);
    });
}

/// Last-writer-wins (no OCC): two concurrent commits must still leave the
/// version monotonic at exactly 2 — the atomic read-bump-swap can't lose one
/// or move the counter backwards, regardless of interleaving.
#[test]
fn lww_commits_are_monotonic() {
    loom::model(|| {
        let s = Arc::new(Session::new());

        let handles: Vec<_> = (0..2)
            .map(|_| {
                let s = s.clone();
                // base is ignored when check_occ = false.
                thread::spawn(move || {
                    s.commit(0, false);
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(*s.inner.lock().unwrap().as_ref(), 2);
    });
}

/// A reader taking a snapshot concurrently with a committer must always see a
/// committed version (0 or 1), never a torn value — the swap is atomic.
#[test]
fn snapshot_never_torn_under_commit() {
    loom::model(|| {
        let s = Arc::new(Session::new());

        let sw = s.clone();
        let w = thread::spawn(move || {
            sw.commit(0, false);
        });
        let sr = s.clone();
        let r = thread::spawn(move || {
            let v = sr.version();
            assert!(v <= 1, "snapshot saw an impossible version: {v}");
        });
        w.join().unwrap();
        r.join().unwrap();
        assert_eq!(s.version(), 1);
    });
}
