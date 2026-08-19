// crates/kglite/src/graph/parallel.rs
//
//! Query-path parallelism: the pool every parallel region runs on, and the
//! interrupt wrapper every parallel region polls.
//!
//! # Why a dedicated pool
//!
//! rayon's global pool sizes its worker stacks from the platform default
//! (2 MiB on the targets we ship). The query pipeline recurses once per level
//! of expression nesting in the planner walkers, in `evaluate_expression` and
//! in the AST's recursive `Drop`, and
//! [`crate::graph::session::QUERY_THREAD_STACK_SIZE`] (8 MiB) is the figure
//! `languages::cypher::stack_probe` calibrates the parser's nesting budget
//! against. A Rust stack overflow **aborts the process**, so a deep query
//! that happened to be evaluated on a global-pool worker would take down
//! every other session sharing it — including every connected Bolt client.
//! The Bolt and MCP servers already hand their own query threads
//! `QUERY_THREAD_STACK_SIZE`; this pool extends the same guarantee to the
//! rayon workers those threads fan out onto.
//!
//! Construction is **lazy**: the pool is built on the first
//! [`install`] call, so a workload that never crosses a parallel site's row
//! threshold never pays for it.
//!
//! # Why an interrupt wrapper
//!
//! Every parallel region has to poll the executor's deadline / cancel flag,
//! or a long fan-out becomes uninterruptible — the shape first written for
//! hop expansion (`core::pattern_matching::matcher_expansion`) and now shared
//! by every site through [`ParallelInterrupt`].

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::graph::session::QUERY_THREAD_STACK_SIZE;

/// Overrides the query pool's width. Unset (the default) means
/// [`std::thread::available_parallelism`]. Values that do not parse, or
/// parse to zero, are ignored.
pub(crate) const QUERY_THREADS_ENV: &str = "KGLITE_QUERY_THREADS";

/// Units of work between two interrupt polls inside a parallel region —
/// the same cadence
/// [`languages::cypher::executor::INTERRUPT_POLL_INTERVAL`] uses for
/// sequential hot loops. Must stay a power of two: [`ParallelInterrupt`]
/// gates on a mask.
///
/// [`languages::cypher::executor::INTERRUPT_POLL_INTERVAL`]: crate::graph::languages::cypher::executor
pub(crate) const PARALLEL_POLL_INTERVAL: usize = 4096;

/// `None` only if the pool could not be built (thread-spawn failure under
/// exhaustion). Callers then run the region on the calling thread, which is
/// slower but always correct — a query is never failed over a pool.
static QUERY_POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();

fn configured_width() -> usize {
    if let Some(raw) = std::env::var_os(QUERY_THREADS_ENV) {
        if let Some(n) = raw
            .to_str()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|n| *n > 0)
        {
            return n;
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn pool() -> Option<&'static rayon::ThreadPool> {
    QUERY_POOL
        .get_or_init(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(configured_width())
                .stack_size(QUERY_THREAD_STACK_SIZE)
                .thread_name(|i| format!("kglite-query-{i}"))
                .build()
                .ok()
        })
        .as_ref()
}

/// Run `op` on the dedicated query pool, so any rayon iterator it drives
/// executes on workers with [`QUERY_THREAD_STACK_SIZE`] stacks.
///
/// Building the pool is deferred to the first call, so this is the only
/// entry point — a parallel site that stays below its row threshold must
/// not call it.
pub(crate) fn install<OP, R>(op: OP) -> R
where
    OP: FnOnce() -> R + Send,
    R: Send,
{
    match pool() {
        Some(p) => p.install(op),
        None => op(),
    }
}

/// The message used when a region reported failure without one — only
/// reachable if a caller sets the flag without recording a reason.
const UNKNOWN_REASON: &str = "parallel region failed";

/// First-error latch + interrupt poll shared by every query-path parallel
/// region.
///
/// `probe` returns `Some(reason)` when the region should abort — the
/// executor passes its deadline/cancel check, the pattern matcher passes
/// `interrupt_reason`. Only the *first* reason is kept: later workers see
/// the latch and stop without overwriting it, so the error a query reports
/// does not depend on which worker noticed second.
///
/// Two consumption styles, both supported:
///
/// * short-circuiting iterators (`try_for_each`, `collect::<Result<_, _>>`)
///   call [`check`](Self::check) and `?` — rayon propagates the error and
///   stops the region;
/// * non-short-circuiting iterators (`flat_map` + `collect`) call
///   [`check`](Self::check)/[`capture`](Self::capture) to latch and yield
///   nothing, then [`finish`](Self::finish) after the region.
pub(crate) struct ParallelInterrupt<F> {
    had_error: AtomicBool,
    first_error: Mutex<Option<String>>,
    probe: F,
}

impl<F> ParallelInterrupt<F>
where
    F: Fn() -> Option<String> + Sync,
{
    pub(crate) fn new(probe: F) -> Self {
        ParallelInterrupt {
            had_error: AtomicBool::new(false),
            first_error: Mutex::new(None),
            probe,
        }
    }

    /// Poll at chunk granularity. `index` is the region-global unit index,
    /// so the probe runs once per [`PARALLEL_POLL_INTERVAL`] units however
    /// rayon happens to split the range. Steady-state cost is one relaxed
    /// load plus a mask.
    #[inline]
    pub(crate) fn check(&self, index: usize) -> Result<(), String> {
        if index & (PARALLEL_POLL_INTERVAL - 1) == 0 {
            return self.check_each();
        }
        if self.had_error.load(Ordering::Relaxed) {
            return Err(self.recorded());
        }
        Ok(())
    }

    /// Poll on every unit. For regions whose per-unit work (a hop expansion,
    /// a pattern count) dwarfs the probe's `Instant::now()`.
    #[inline]
    pub(crate) fn check_each(&self) -> Result<(), String> {
        if self.had_error.load(Ordering::Relaxed) {
            return Err(self.recorded());
        }
        if let Some(reason) = (self.probe)() {
            self.fail(reason.clone());
            return Err(reason);
        }
        Ok(())
    }

    /// Latch `reason` if nothing has failed yet.
    pub(crate) fn fail(&self, reason: String) {
        if !self.had_error.swap(true, Ordering::Relaxed) {
            *self.first_error.lock().unwrap() = Some(reason);
        }
    }

    /// Latch a fallible unit's error and yield `None` in its place — for
    /// `flat_map`/`map` regions that cannot return `Result`.
    #[inline]
    pub(crate) fn capture<T>(&self, outcome: Result<T, String>) -> Option<T> {
        match outcome {
            Ok(value) => Some(value),
            Err(reason) => {
                self.fail(reason);
                None
            }
        }
    }

    fn recorded(&self) -> String {
        self.first_error
            .lock()
            .unwrap()
            .clone()
            .unwrap_or_else(|| UNKNOWN_REASON.to_string())
    }

    /// Consume the latch after the region: `Err` with the first reason if
    /// any worker aborted.
    pub(crate) fn finish(self) -> Result<(), String> {
        if self.had_error.load(Ordering::Relaxed) {
            return Err(self
                .first_error
                .into_inner()
                .unwrap()
                .unwrap_or_else(|| UNKNOWN_REASON.to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;

    #[test]
    fn poll_interval_is_a_power_of_two() {
        assert!(PARALLEL_POLL_INTERVAL.is_power_of_two());
    }

    #[test]
    fn pool_workers_get_the_query_thread_stack_size() {
        let width = install(rayon::current_num_threads);
        assert!(width >= 1);
        // `install` must run the closure *on* the pool, not inline on the
        // caller — otherwise the stack guarantee it exists for is void.
        let name = install(|| {
            (0..1024usize)
                .into_par_iter()
                .map(|_| {
                    std::thread::current()
                        .name()
                        .unwrap_or("<unnamed>")
                        .to_string()
                })
                .find_any(|n| n.starts_with("kglite-query-"))
        });
        assert!(
            name.is_some(),
            "parallel work inside install() did not run on a kglite-query worker"
        );
    }

    #[test]
    fn first_reason_wins_and_later_workers_see_the_latch() {
        let guard = ParallelInterrupt::new(|| None);
        guard.fail("first".to_string());
        guard.fail("second".to_string());
        assert_eq!(guard.check_each(), Err("first".to_string()));
        assert_eq!(guard.finish(), Err("first".to_string()));
    }

    #[test]
    fn check_polls_only_on_chunk_boundaries() {
        let polls = AtomicBool::new(false);
        let guard = ParallelInterrupt::new(|| {
            polls.store(true, Ordering::Relaxed);
            None
        });
        assert_eq!(guard.check(1), Ok(()));
        assert!(!polls.load(Ordering::Relaxed), "off-boundary index polled");
        assert_eq!(guard.check(PARALLEL_POLL_INTERVAL), Ok(()));
        assert!(polls.load(Ordering::Relaxed), "boundary index did not poll");
    }

    #[test]
    fn capture_latches_a_unit_error() {
        let guard = ParallelInterrupt::new(|| None);
        assert_eq!(guard.capture(Ok(7)), Some(7));
        assert_eq!(guard.capture::<i32>(Err("boom".into())), None);
        assert_eq!(guard.finish(), Err("boom".to_string()));
    }
}
