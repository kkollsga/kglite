//! The Cypher engine's abort check — one poll, two reportable outcomes.
//!
//! Both engines route their deadline/cancel polls through here: the read
//! executor via `CypherExecutor::check_deadline`, the mutation engine via
//! `write::check_interrupt_periodic`. Keeping the wording in one place is what
//! lets the session layer classify the abort — a passed deadline is
//! [`KgError::CypherTimeout`], a raised cancel flag is [`KgError::Cancelled`] —
//! off the interrupt state rather than off this prose.
//!
//! [`KgError::CypherTimeout`]: crate::error::KgError::CypherTimeout
//! [`KgError::Cancelled`]: crate::error::KgError::Cancelled

use crate::graph::algorithms::Interrupt;

/// Report which half of `interrupt` fired, if either.
///
/// The mutation engine used to run its own poller reporting a flat "Query
/// interrupted" for both, which is how a plain deadline on a write reached
/// callers as an unclassifiable execution failure.
#[inline]
pub(super) fn check_interrupt(interrupt: &Interrupt) -> Result<(), String> {
    if interrupt.deadline_expired() {
        return Err(TIMEOUT_MESSAGE.to_string());
    }
    if interrupt.is_cancelled() {
        return Err(CANCELLED_MESSAGE.to_string());
    }
    Ok(())
}

/// Wording for a fired deadline. Read by callers as prose only — the
/// timeout-vs-execution-failure classification is structural
/// (`session::execute::exec_err` re-reads the interrupt state), never a match
/// on this text.
const TIMEOUT_MESSAGE: &str = "Query timed out. Hints: anchor the query with MATCH (n {id: ...}) \
     or a pattern property matching an indexed column (e.g. \
     MATCH (n {label: 'X'})). To allow a longer run, pass \
     timeout_ms=N to cypher() or set kg.set_default_timeout(ms); \
     timeout_ms=0 disables the deadline.";

/// Wording for a raised cancel flag; matches the pattern matcher's
/// (`pattern_matching::matcher`) so both carriers read identically.
const CANCELLED_MESSAGE: &str = "Query cancelled";
