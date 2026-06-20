//! Cross-cutting helpers for the PyO3 boundary.
//!
//! [`EnterKg`] is the single chokepoint for "release the GIL, run a
//! blocking chunk of engine work, observe Ctrl-C, and map any
//! [`KgError`] back to the typed Python exception". Before this existed,
//! every `#[pymethods]` body open-coded
//! `py.detach(move || ...).map_err(crate::error_py::kg_to_pyerr)`, which
//! made it easy to forget the GIL release or the error mapping and left
//! no single place to add cancellation.
//!
//! Mirrors the role polars' `EnterPolarsExt::enter_polars`
//! (`crates/polars-python/src/utils.rs`) plays in that binding.
//!
//! ## Ctrl-C / interruptible queries
//!
//! While the closure runs (GIL released), a scoped SIGINT handler is
//! installed that flips a process-global `AtomicBool`. The engine polls
//! that flag at the same checkpoints it polls the query deadline (see
//! `ExecuteOptions::cancel`), so a long `MATCH`/scan aborts promptly with
//! `KgError::Cancelled`, which maps to Python's `KeyboardInterrupt`. The
//! previous SIGINT disposition (normally Python's own handler) is
//! restored when the call returns.
//!
//! **Scope:** this targets the interactive single-query case
//! (notebook / REPL on the main thread). The flag and handler are
//! process-global, so under concurrent multi-thread `Session` serving a
//! Ctrl-C cancels whichever query last entered `enter_kg`; servers run
//! their own signal model and don't use this path. The handler is
//! Unix-only — on other platforms the deadline still bounds queries but
//! Ctrl-C is a no-op mid-query.

use pyo3::marker::Ungil;
use pyo3::prelude::*;
use std::sync::atomic::AtomicBool;

use crate::error::KgError;
use crate::error_py::kg_to_pyerr;

/// GIL-release + cancellation + error-mapping helper on [`Python`].
///
/// Wrap any blocking engine call so the GIL is released for its
/// duration, Ctrl-C is observable, and a returned [`KgError`] becomes
/// the most specific `kglite.*` Python exception.
pub(crate) trait EnterKg {
    /// Release the GIL, install the scoped SIGINT handler, run `f`
    /// (handing it the cooperative-cancel flag to thread into
    /// `ExecuteOptions::cancel`), then map `Err(e)` through
    /// [`kg_to_pyerr`]. The closure receives `None` for the cancel flag
    /// on platforms without the handler, in which case it should still
    /// pass it straight through — the engine treats `None` as
    /// "never cancelled".
    fn enter_kg<T, E, F>(self, f: F) -> PyResult<T>
    where
        F: Ungil + Send + FnOnce(Option<&'static AtomicBool>) -> Result<T, E>,
        T: Ungil + Send,
        E: Ungil + Send + Into<KgError>;
}

impl EnterKg for Python<'_> {
    #[inline]
    fn enter_kg<T, E, F>(self, f: F) -> PyResult<T>
    where
        F: Ungil + Send + FnOnce(Option<&'static AtomicBool>) -> Result<T, E>,
        T: Ungil + Send,
        E: Ungil + Send + Into<KgError>,
    {
        let _guard = sigint::install();
        let cancel = sigint::cancel_flag();
        // GIL released for the whole engine call; the SIGINT handler (set
        // by `_guard`) can flip `cancel` from signal context.
        let r = self.detach(move || f(cancel));
        // `_guard` restores the prior SIGINT handler on drop (after the
        // GIL is reacquired here).
        r.map_err(|e| kg_to_pyerr(e.into()))
    }
}

/// Scoped SIGINT (Ctrl-C) handling. On Unix, installs a handler for the
/// duration of an `enter_kg` call that flips a process-global flag the
/// engine polls; restores the previous handler when the last active call
/// finishes. On other platforms every operation is a no-op.
#[cfg(unix)]
mod sigint {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;

    /// Process-global cancel flag. The engine polls it via
    /// `ExecuteOptions::cancel`; the signal handler sets it.
    static QUERY_CANCEL: AtomicBool = AtomicBool::new(false);

    /// `Some(&QUERY_CANCEL)` — handed to `ExecuteOptions::cancel`.
    pub(super) fn cancel_flag() -> Option<&'static AtomicBool> {
        Some(&QUERY_CANCEL)
    }

    /// Async-signal-safe: a single relaxed atomic store. No allocation,
    /// no lock, no reentrancy hazard.
    extern "C" fn handle_sigint(_sig: libc::c_int) {
        QUERY_CANCEL.store(true, Ordering::SeqCst);
    }

    struct State {
        /// Number of `enter_kg` calls currently holding the handler.
        depth: usize,
        /// SIGINT disposition to restore when `depth` returns to 0
        /// (normally Python's own handler).
        prev: libc::sigaction,
    }

    static STATE: Mutex<Option<State>> = Mutex::new(None);

    pub(super) struct Guard;

    /// Install the handler (refcounted across concurrent/nested calls)
    /// and clear any stale cancel flag from a prior run. Returns a guard
    /// that restores the previous handler on drop.
    pub(super) fn install() -> Guard {
        let mut g = STATE.lock().unwrap_or_else(|e| e.into_inner());
        match g.as_mut() {
            Some(state) => state.depth += 1,
            None => {
                // First active call: clear stale cancel, save the current
                // SIGINT handler, install ours.
                QUERY_CANCEL.store(false, Ordering::SeqCst);
                // SAFETY: zeroed sigaction is a valid empty mask + flags;
                // we then set the handler field. `sigaction` is the POSIX
                // install call.
                let mut new: libc::sigaction = unsafe { std::mem::zeroed() };
                // Cast through an explicit fn-pointer type first (avoids the
                // `fn_to_numeric_cast` lint on a direct function-item cast).
                let handler = handle_sigint as extern "C" fn(libc::c_int);
                new.sa_sigaction = handler as libc::sighandler_t;
                let mut prev: libc::sigaction = unsafe { std::mem::zeroed() };
                unsafe {
                    libc::sigaction(libc::SIGINT, &new, &mut prev);
                }
                *g = Some(State { depth: 1, prev });
            }
        }
        Guard
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            let mut g = STATE.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(state) = g.as_mut() {
                state.depth -= 1;
                if state.depth == 0 {
                    // SAFETY: `prev` was captured from a prior successful
                    // `sigaction` call and is a valid disposition.
                    unsafe {
                        libc::sigaction(libc::SIGINT, &state.prev, std::ptr::null_mut());
                    }
                    *g = None;
                }
            }
        }
    }
}

#[cfg(not(unix))]
mod sigint {
    use std::sync::atomic::AtomicBool;

    pub(super) fn cancel_flag() -> Option<&'static AtomicBool> {
        None
    }

    pub(super) struct Guard;

    pub(super) fn install() -> Guard {
        Guard
    }
}
