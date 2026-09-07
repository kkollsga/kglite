//! Captured per-surface query policy: the deadline (and its default), the work
//! budget and the row cap a binding applies when a call omits them.
//!
//! This lives in the core, not in a binding, because the *policy value* must
//! not be re-typed per surface. It began as `kglite-py`'s private
//! `query_defaults.rs`; the MCP server then needed the same 180 s deadline for
//! a liveness reason of its own (an agent has no cancel channel, and one
//! runaway read stalls the reload gate every later tool call goes through), and
//! a second copy of `180_000` is exactly how two surfaces come to disagree
//! about what "the default" is. Which surfaces *adopt* the default is still a
//! per-binding decision — the CLI deliberately declares none, because a human
//! at a terminal has Ctrl-C and a batch query over a Wikidata-scale graph
//! legitimately runs for hours.

use std::time::{Duration, Instant};

/// The deadline a surface that adopts a default applies to a query that names
/// none: three minutes. Python and the MCP server adopt it; the CLI and the
/// Bolt server declare that they do not (see `docs/operators/cli.md` and
/// `docs/operators/bolt-server.md`).
pub const DEFAULT_TIMEOUT_MS: u64 = 180_000;

/// Per-handle query policy, captured when a handle is derived from another.
///
/// Every field is `Option`, and `None` means "inherit" rather than "off" —
/// `Some(0)` is what turns a knob off, which is why `resolve` reads the two
/// differently.
#[derive(Clone, Copy, Debug, Default)]
pub struct QueryDefaults {
    pub timeout_ms: Option<u64>,
    pub max_work_units: Option<usize>,
    pub row_limit: Option<usize>,
}

/// What one call actually runs under, after the call's own arguments have been
/// laid over the captured defaults.
pub struct ResolvedQueryOptions {
    pub timeout_ms: Option<u64>,
    pub deadline: Option<Instant>,
    pub max_work_units: Option<usize>,
    pub row_limit: Option<usize>,
}

/// A deadline from an explicit millisecond budget. `0` disables the deadline
/// on every surface, and so does `None` — a caller that wants the default
/// applied goes through [`QueryDefaults::resolve`], which supplies it.
pub fn deadline_from(timeout_ms: Option<u64>) -> Option<Instant> {
    timeout_ms
        .filter(|ms| *ms != 0)
        .map(|ms| Instant::now() + Duration::from_millis(ms))
}

impl QueryDefaults {
    /// Lay one call's arguments over the captured policy, falling back to
    /// [`DEFAULT_TIMEOUT_MS`] when neither names a deadline.
    pub fn resolve(
        self,
        timeout_ms: Option<u64>,
        max_work_units: Option<usize>,
        row_limit: Option<usize>,
    ) -> ResolvedQueryOptions {
        let timeout_ms = timeout_ms.or(self.timeout_ms).or(Some(DEFAULT_TIMEOUT_MS));
        ResolvedQueryOptions {
            timeout_ms: timeout_ms.filter(|ms| *ms != 0),
            deadline: deadline_from(timeout_ms),
            max_work_units: max_work_units.or(self.max_work_units),
            row_limit: row_limit.or(self.row_limit),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_zero_and_optional_inheritance_have_distinct_meanings() {
        let policy = QueryDefaults {
            timeout_ms: Some(0),
            max_work_units: Some(1),
            row_limit: Some(2),
        };
        let inherited = policy.resolve(None, None, None);
        assert!(inherited.deadline.is_none());
        assert_eq!(inherited.max_work_units, Some(1));
        assert_eq!(inherited.row_limit, Some(2));
        let explicit = policy.resolve(Some(5), Some(0), Some(0));
        assert!(explicit.deadline.is_some());
        assert_eq!(explicit.max_work_units, Some(0));
        assert_eq!(explicit.row_limit, Some(0));
        assert!(deadline_from(Some(0)).is_none());
    }

    /// The default is the whole point of the lift: a surface that adopts it and
    /// names no deadline runs under three minutes, not forever.
    #[test]
    fn an_unset_policy_resolves_to_the_shared_default() {
        let resolved = QueryDefaults::default().resolve(None, None, None);
        assert_eq!(resolved.timeout_ms, Some(DEFAULT_TIMEOUT_MS));
        assert_eq!(DEFAULT_TIMEOUT_MS, 180_000);
        let deadline = resolved.deadline.expect("the default supplies a deadline");
        let remaining = deadline.saturating_duration_since(Instant::now());
        assert!(
            remaining <= Duration::from_millis(DEFAULT_TIMEOUT_MS)
                && remaining > Duration::from_millis(DEFAULT_TIMEOUT_MS - 5_000),
            "expected roughly {DEFAULT_TIMEOUT_MS} ms out, got {remaining:?}"
        );
    }
}
