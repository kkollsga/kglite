//! Graph algorithms — the scoring / analytics side of the codebase.
//!
//! PageRank, centrality, components, shortest path, clustering, vector
//! search. Distinguishes from `query/` which answers "which nodes match
//! this pattern?"; algorithms here answer "what does the graph look
//! like structurally?".

pub mod centrality;
pub mod clustering;
pub mod community;
pub mod graph_algorithms;
pub mod hnsw;
pub mod vector;

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Deadline + cooperative-cancellation bundle threaded through the graph
/// algorithms, replacing the bare `deadline: Option<Instant>` they used to
/// take. `Copy` + `Send` + `Sync` so it rides rayon closures for free.
///
/// `exceeded()` is the single abort check: it fires when the query deadline
/// has passed *or* a binding flipped the cancel flag (the Python wheel's
/// Ctrl-C / SIGINT handler). The cancel flag is `&'static` because its only
/// setter is a process-global signal handler — see
/// [`crate::api::session::ExecuteOptions::cancel`].
#[derive(Clone, Copy, Default)]
pub struct Interrupt {
    pub deadline: Option<Instant>,
    pub cancel: Option<&'static AtomicBool>,
}

impl Interrupt {
    /// Deadline-only interrupt (no cancellation wired) — the shape callers
    /// that don't thread a cancel flag (tests, internal helpers) want.
    #[inline]
    pub fn from_deadline(deadline: Option<Instant>) -> Self {
        Self {
            deadline,
            cancel: None,
        }
    }

    /// `true` when the run should abort: deadline passed or cancel flagged.
    /// One `Instant::now()` (only if a deadline is set) plus one relaxed
    /// atomic load (only if a cancel flag is set); both `None` → no cost.
    #[inline]
    pub fn exceeded(&self) -> bool {
        self.deadline.is_some_and(|dl| Instant::now() > dl)
            || self.cancel.is_some_and(|c| c.load(Ordering::Relaxed))
    }

    /// Whether a finite deadline is set (some algorithms gate scoped-vs-
    /// streaming routing on deadline presence).
    #[inline]
    pub fn is_bounded(&self) -> bool {
        self.deadline.is_some()
    }
}

impl From<Option<Instant>> for Interrupt {
    #[inline]
    fn from(deadline: Option<Instant>) -> Self {
        Self::from_deadline(deadline)
    }
}
